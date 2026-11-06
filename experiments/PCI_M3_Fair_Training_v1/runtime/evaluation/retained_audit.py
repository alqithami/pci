"""Reconcile a complete closed M2 evaluation from retained records.

This replays the independent scalar effect and the recorded policy RNG against
stored states. It does not construct a new task world, execute a native task
transition, train, query a GPU or call a remote service. The portable actor is
independent of the authoritative CPU Torch forward pass used during evaluation.
"""
from __future__ import annotations
import argparse, copy, gzip, json, math, time
from pathlib import Path
import numpy as np
from .common import require, digest, sha, read_json, write_new_json, local
from .actor import load_export, portable_forward, LOGIT_ATOL, PROBABILITY_ATOL
from .engine import PROFILES, METHODS, COUNT_KEYS, checked_adapter, validate_plan, source_closure, physical_actions_scalar, training_work
from .reference import scalar_world as scalar


def static_features(snapshot, adapter):
    """Restore fields directly, with no constructor, reset, transition or RNG."""
    s = scalar.validate_snapshot(snapshot)
    env = object.__new__(adapter.mod.Warehouse)
    env.config = adapter.mod.EnvConfig(**s['config'])
    env.walls, env.regions, env.pickups, env.depots, env.floor, env.distances = adapter.mod.geometry(env.config.layout)
    env.episode, env.t, env.last = s['episode'], s['t'], copy.deepcopy(s['last'])
    for k in ('pos','counts','delivered','wait','tasks','destinations','task_index','goals'):
        setattr(env, k, np.array(s[k], dtype=np.int64))
    env.carrying = np.array(s['carrying'], dtype=bool)
    class ForbiddenRNG:
        def __getattr__(self, name):
            raise ValueError('STATIC_FEATURE_RNG_FORBIDDEN:' + name)
    env.rng = ForbiddenRNG()
    return env.features()


def decode_line(raw):
    require(len(raw) <= 2*1024*1024 and raw.endswith(b'\n'), 'TRACE_LINE_SIZE_OR_NEWLINE')
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'DUPLICATE_TRACE_KEY')
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('NONFINITE_TRACE_JSON')))


def independent_counts(effect, physical):
    """Recount actual stage distances; quota flags are not repair counts."""
    proposed, admitted = effect['proposal'], effect['admitted_actions']
    p = sum(proposed[a] != physical[a] for a in range(4))
    i = sum(physical[a] != admitted[a] for a in range(4))
    t = sum(proposed[a] != admitted[a] for a in range(4))
    m = effect['result']['metrics']
    wall, collision, quota, excess = (m['wall_attempts'],m['collision_attempts'],m['quota_violations'],m['capacity_excess'])
    require(t == m['interventions'] and abs(p-i) <= t <= p+i, 'STAGE_DISTANCE_ACCOUNTING')
    require(m['proposed_compliant'] == int(not(wall or collision or quota or excess)), 'PROPOSAL_COMPLIANCE_ACCOUNTING')
    return dict(physical_agent_replacements=p, institutional_stage_agent_replacements=i,
        total_agent_replacements=t, physical_bad_joint_steps=int(wall > 0 or collision > 0),
        institutional_bad_joint_steps=int(quota > 0 or excess > 0),
        quota_bad_joint_steps=int(quota > 0),capacity_bad_joint_steps=int(excess > 0),
        proposed_compliant_joint_steps=m['proposed_compliant'],deliveries=m['delivered'],pickups=m['pickups'],
        wall_agent_attempts=wall,collision_agent_attempts=collision,raw_quota_violating_agent_decisions=quota,
        raw_capacity_excess=excess,executed_compliant_joint_steps=m['executed_compliant'],
        moved_agent_decisions=m['move_count'],nonadditive_stage_steps=int(p+i != t))


def verify_closed_inventory(output):
    root = Path(output).resolve()
    require(not (root/'EVALUATION_FAILED.json').exists(), 'FAILED_COHORT_HAS_NO_ANALYSIS')
    manifest = read_json(root/'EVALUATION_COMPLETE.json')
    require(manifest.get('schema') == 'PCI-M2-EVALUATION-CLOSED-MANIFEST-v1' and manifest.get('status') == 'PASS' and manifest.get('scientific') is True, 'COMPLETE_SCIENTIFIC_COHORT_REQUIRED')
    require(manifest['complete_cases'] == 1200 and manifest['joint_steps'] == 307200 and manifest['agent_decisions'] == 1228800, 'EXACT_PRIMARY_EVALUATION_COUNTS')
    files = manifest['files']
    require(type(files) is dict and len(files) == 2401, 'EXACT_CLOSED_PRIMARY_FILE_COUNT')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == set(files) | {'EVALUATION_COMPLETE.json'}, 'NO_UNMANIFESTED_OR_MISSING_PRIMARY_FILES')
    for n, item in files.items():
        p = local(root, n)
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == item['bytes'] > 0 and sha(p) == item['sha256'], 'CLOSED_PRIMARY_FILE_IDENTITY:' + n)
    return manifest


def expected_initial_snapshot(cfg, seed):
    """Independent replay of the declared native initialization RNG operations.

    No Warehouse object, features, task transition or native reset is called.
    These are explicitly counted old-input RNG replay streams, not new episodes.
    """
    cfg = scalar.config(cfg)
    n, horizon = cfg['n_agents'], cfg['horizon']
    walls, regions = scalar.geometry(cfg['layout'])
    floor = [(x,y) for y in range(13) for x in range(13) if (x,y) not in walls]
    allowed = [p for p in floor if not any(scalar._inside(p,r) for r in regions)]
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(allowed), n, replace=False)
    positions = [list(allowed[int(k)]) for k in selected]
    tasks = rng.integers(0,len(scalar.PICKUPS),size=(n,horizon+2)).tolist()
    destinations = rng.integers(0,len(scalar.DEPOTS),size=(n,horizon+2)).tolist()
    return dict(config=cfg,episode=0,t=0,pos=positions,carrying=[False]*n,
        delivered=[0]*n,wait=[0]*n,counts=[[0,0] for _ in range(n)],
        tasks=tasks,destinations=destinations,task_index=[0]*n,
        goals=[list(scalar.PICKUPS[tasks[i][0]]) for i in range(n)],last=None,
        rng_state=copy.deepcopy(rng.bit_generator.state))


def audit_case(output, case, binding, arrays, adapter, protocol_sha256, *, fixture=False, expected_config=None):
    root = Path(output)
    summary = read_json(local(root, f"cases/{case['case_id']}.json"))
    require(summary['status'] == 'PASS' and summary['scientific'] is (not fixture) and summary['case'] == case, 'COMPLETE_CASE_IDENTITY')
    horizon = case['horizon']
    require(type(horizon) is int and 2 <= horizon <= 256, 'AUDIT_CASE_HORIZON')
    cfg = PROFILES[case['profile_id']] if expected_config is None else scalar.config(expected_config)
    require(fixture or (horizon == 256 and cfg == PROFILES[case['profile_id']]), 'SCIENTIFIC_CASE_PROFILE_REQUIRED')
    require(summary['config'] == cfg and summary['joint_steps'] == horizon and summary['agent_decisions'] == 4*horizon, 'CASE_CONFIG_AND_UNITS')
    trace = local(root, summary['trace_path'])
    require(summary['trace_path'] == f"traces/{case['case_id']}.jsonl.gz" and sha(trace) == summary['trace_sha256'] and trace.stat().st_size == summary['trace_bytes'] > 0, 'CASE_TRACE_IDENTITY')
    total = dict.fromkeys(COUNT_KEYS, 0)
    reward_sum = 0.0
    def empty(**labels):
        return dict(**labels,joint_steps=0,agent_decisions=0,counts=dict.fromkeys(COUNT_KEYS,0),task_reward_sum=0.0)
    phases=[empty(phase=p) for p in range(cfg['quota_window'])]
    halves=[empty(half=h) for h in range(2)]
    last = first = None
    max_errors = {'logit_abs_error':0.0,'probability_abs_error':0.0}
    rng = np.random.default_rng(case['policy_sampling_seed'])
    with gzip.open(trace, 'rb') as f:
        raw_header = f.readline(2*1024*1024+1)
        require(bool(raw_header), 'NONEMPTY_COMPLETE_TRACE')
        h = decode_line(raw_header)
        header, chain = h['header'], h['chain']
        require(header['schema'] == 'PCI-M2-EVALUATION-TRACE-HEADER-v1' and header['scientific'] is (not fixture) and header['case'] == case, 'SCIENTIFIC_TRACE_HEADER')
        require(header['protocol_sha256'] == protocol_sha256 and header['checkpoint_sha256'] == binding['checkpoint_sha256'] and header['portable_export_sha256'] == binding['portable_export_sha256'] and header['model_signature'] == binding['signature'], 'TRACE_INPUT_BINDING')
        require(chain == digest(header) == summary['trace_header_sha256'], 'TRACE_HEADER_CHAIN')
        for t in range(horizon):
            line = f.readline(2*1024*1024+1)
            require(bool(line), 'MISSING_STEP_RECORD')
            r = decode_line(line)
            p = r['payload']
            require(r['previous'] == chain and p['case_id'] == case['case_id'] and p['step'] == t, 'CONTIGUOUS_TRACE_CHAIN')
            require(r['chain'] == digest({'previous':chain,'payload':p}), 'TRACE_RECORD_DIGEST')
            effect = p['effect']
            before = effect['before']
            require(before['t'] == t and before['config'] == summary['config'], 'RETAINED_STEP_AND_CONFIG')
            if last is not None:
                require(scalar.canonical(before) == scalar.canonical(last), 'RETAINED_WORLD_CONTINUITY')
            else:
                require(before['t'] == 0, 'EPISODE_START_AT_ZERO')
                require(scalar.canonical(before) == scalar.canonical(expected_initial_snapshot(cfg, case['world_seed'])), 'INITIAL_WORLD_SEED_RNG_REPLAY')
                first = copy.deepcopy(before)
            # Independent scalar replay on retained states, without native execution.
            expected = scalar.scalar_effect(before, effect['proposal'])
            require(scalar.canonical(effect) == scalar.canonical(expected), 'RETAINED_SCALAR_FULL_EFFECT')
            physical = physical_actions_scalar(before, effect['proposal'])
            require(physical == p['physical_stage_actions'], 'RETAINED_PHYSICAL_STAGE')
            counts = independent_counts(expected, physical)
            require(counts == p['stage_counts'], 'RETAINED_STAGE_COUNTS')
            for k in COUNT_KEYS:
                require(type(counts[k]) is int and counts[k] >= 0, 'NONNEGATIVE_INTEGER_COUNTS')
                total[k] += counts[k]
            policy = p['policy']
            require(rng.bit_generator.state == policy['rng_before'], 'POLICY_RNG_CONTINUITY')
            u = rng.random(4)
            require(np.array_equal(u, np.asarray(policy['uniforms'])) and rng.bit_generator.state == policy['rng_after'], 'POLICY_RNG_DRAW_ACCOUNTING')
            probabilities = np.asarray(policy['probabilities'], dtype=np.float64)
            logits = np.asarray(policy['logits'], dtype=np.float64)
            require(probabilities.shape == logits.shape == (4,5) and np.isfinite(probabilities).all() and np.isfinite(logits).all() and (probabilities >= 0).all(), 'FINITE_RETAINED_POLICY')
            portable_logits, portable_probabilities = portable_forward(arrays, static_features(before, adapter))
            le = float(np.max(np.abs(logits-portable_logits.astype(np.float64))))
            pe = float(np.max(np.abs(probabilities-portable_probabilities.astype(np.float64))))
            require(le <= LOGIT_ATOL and pe <= PROBABILITY_ATOL, 'INDEPENDENT_RETAINED_ACTOR_FORWARD')
            require(abs(le-p['forward_parity']['logit_abs_error']) <= 1e-15 and abs(pe-p['forward_parity']['probability_abs_error']) <= 1e-15, 'FORWARD_CHECK_ARITHMETIC')
            require(np.allclose(probabilities.sum(axis=-1), 1.0, atol=2e-6, rtol=0), 'RETAINED_SIMPLEX')
            sampling = probabilities / probabilities.sum(axis=-1, keepdims=True)
            cumulative = np.cumsum(sampling, axis=-1)
            # Independent inverse-CDF arithmetic, same declared float64 sum convention.
            actions = [int(np.count_nonzero(u[a] >= cumulative[a,:-1])) for a in range(4)]
            require(actions == effect['proposal'], 'RETAINED_PROPOSAL_SAMPLING')
            reward = expected['result']['metrics']['task_reward']
            reward_sum += reward
            for stratum in (phases[t % cfg['quota_window']],halves[int(t >= horizon//2)]):
                stratum['joint_steps'] += 1;stratum['agent_decisions'] += 4
                for k in COUNT_KEYS:stratum['counts'][k] += counts[k]
                stratum['task_reward_sum'] += reward
            for k, v in [('logit_abs_error', le), ('probability_abs_error', pe)]:
                max_errors[k] = max(max_errors[k], v)
            last, chain = expected['after'], r['chain']
            require(effect['result']['done'] is (t == horizon-1), 'TRUE_FINITE_HORIZON_DONE')
        require(f.readline(1) == b'', 'EXTRA_TRACE_RECORDS')
    require(last['t'] == horizon and last['episode'] == first['episode'], 'FULL_RETAINED_EPISODE')
    require(chain == summary['final_chain'] and scalar.digest(first) == summary['initial_world_sha256'] and scalar.digest(last) == summary['final_world_sha256'], 'CASE_WORLD_ENDPOINT_BINDING')
    require(phases == summary['phase_summaries'] and halves == summary['half_summaries'], 'PHASE_AND_HALF_SUMMARY_RECONCILIATION')
    require(total == summary['counts'] and abs(reward_sum-summary['task_reward_sum']) <= 1e-12 and max_errors == summary['forward_parity_max'], 'CASE_SUMMARY_RECONCILIATION')
    require(total['deliveries'] == sum(last['delivered'])-sum(first['delivered']) and total['executed_compliant_joint_steps'] == horizon, 'RETAINED_TASK_AND_ADMISSION_COUNTS')
    require(sha(trace) == summary['trace_sha256'], 'TRACE_CHANGED_DURING_AUDIT')
    return summary


def resolve_audit_models(training_root, training_cases, bindings):
    """Validate supplied retained models without changing stored bindings.

    The returned absolute locators are operational inputs. Their checkpoint,
    config, completion, closure and portable bodies remain bound to provenance.
    A copied training root is allowed; no original absolute path is required.
    """
    training_root = Path(training_root).resolve()
    require(set(bindings) == {c['case_id'] for c in training_cases}, 'ALL_CLOSED_TRAINING_INPUTS')
    models = {}
    for c in training_cases:
        b = bindings[c['case_id']]
        folder = local(training_root, c['case_id'])
        checkpoint, export = folder/'checkpoint.pt', folder/'final_export.npz'
        require(sha(checkpoint) == b['checkpoint_sha256'] and sha(folder/'TRAIN_COMPLETE.json') == b['train_complete_sha256'] and sha(folder/'config.json') == b['config_sha256'], 'AUDIT_TRAINING_MODEL_INPUTS')
        require(sha(export) == b['portable_export_sha256'], 'AUDIT_EXPORTED_MODEL_INPUT')
        done=read_json(folder/'TRAIN_COMPLETE.json');config=read_json(folder/'config.json')
        require(config['method'] == c['method'] and config['seed'] == c['training_seed'], 'AUDIT_ACTUAL_CONFIG_IDENTITY')
        require(done['method'] == c['method'] and done['seed'] == c['training_seed'] and done['signature'] == b['signature'], 'AUDIT_ACTUAL_MODEL_SIGNATURE')
        require(sha(folder/'CLOSED_FILES.json') == b['closed_training_manifest_sha256'] == done['closed_manifest_sha256'] and training_work(config,done) == b['training_work'], 'AUDITED_TRAINING_WORK_METADATA')
        load_export(export)
        models[c['case_id']] = dict(training_case_id=c['case_id'],method=c['method'],training_seed=c['training_seed'],
            signature=done['signature'],portable_export_sha256=b['portable_export_sha256'],export_path=str(export),
            checkpoint_path=str(checkpoint),checkpoint_sha256=b['checkpoint_sha256'],
            config_path=str(folder/'config.json'),config_sha256=b['config_sha256'],
            train_complete_path=str(folder/'TRAIN_COMPLETE.json'),train_complete_sha256=b['train_complete_sha256'])
    return models


def audit(project_root, training_root, protocol_path, output, receipt, *, operations_root, deadline_monotonic):
    start = time.perf_counter()
    root, training_root, output, receipt = map(lambda p: Path(p).resolve(), (project_root,training_root,output,receipt))
    require(not receipt.exists() and not receipt.is_relative_to(output), 'NEW_EXTERNAL_AUDIT_RECEIPT_REQUIRED')
    protocol = read_json(protocol_path)
    training, cases, _ = validate_plan(protocol)
    manifest = verify_closed_inventory(output)
    manifest_sha = sha(output/'EVALUATION_COMPLETE.json')
    protocol_sha = sha(protocol_path)
    require(manifest['protocol_sha256'] == protocol_sha, 'AUDITED_PROTOCOL_IDENTITY')
    provenance = read_json(output/'PROVENANCE.json')
    require(provenance['protocol_sha256'] == protocol_sha and provenance['source_sha256'] == source_closure(root), 'AUDITED_SOURCE_CLOSURE')
    bindings = provenance['model_bindings']
    require(set(bindings) == {c['case_id'] for c in training}, 'ALL_CLOSED_TRAINING_INPUTS')
    validated_models = resolve_audit_models(training_root,training,bindings)
    export_paths = {key: model['export_path'] for key, model in validated_models.items()}
    adapter = checked_adapter(root)
    from .parallel import run_phase
    phase = run_phase(root,output,cases,bindings,protocol_sha,'audit',operations_root,deadline_monotonic,
                      expected_source_sha256=provenance['source_sha256'],
                      export_paths=export_paths,validated_models=validated_models)
    summaries = phase['summaries']
    require(len(summaries) == 1200 and sum(s['joint_steps'] for s in summaries) == 307200, 'ALL_1200_INDEPENDENTLY_RECONCILED')
    require(source_closure(root) == provenance['source_sha256'], 'SOURCE_CHANGED_DURING_RETAINED_AUDIT')
    require(sha(output/'EVALUATION_COMPLETE.json') == manifest_sha, 'CLOSED_MANIFEST_CHANGED_DURING_AUDIT')
    result = dict(schema='PCI-M2-EVALUATION-RETAINED-AUDIT-v1', status='PASS',
        protocol_sha256=protocol_sha, closed_manifest_sha256=manifest_sha, checked_cases=1200,
        checked_joint_steps=307200, checked_agent_decisions=1228800, model_inputs=60,
        source_sha256=source_closure(root), elapsed_seconds=time.perf_counter()-start,
        new_native_task_worlds=0, native_task_transition_calls=0, training_GPU_proof_cloud_calls=0,
        policy_rng_streams_replayed=1200, world_initialization_rng_streams_replayed=1200,
        scope='Reconciles closed-file hashes, complete retained scalar effects, physical/institutional Hamming stages, independent portable F32 actor outputs, recorded PCG64 sampling, independent initialization-seed RNG replay and all summaries. Uses stored states; no new task world, native transition, new independent training, second autodiff or cryptographic assurance.')
    write_new_json(receipt, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','training-root','protocol','output','receipt'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--operations-root',type=Path,required=True)
    p.add_argument('--deadline-monotonic',type=float,required=True)
    a = p.parse_args()
    r = audit(a.project_root,a.training_root,a.protocol,a.output,a.receipt,
              operations_root=a.operations_root,deadline_monotonic=a.deadline_monotonic)
    print(json.dumps({k:r[k] for k in ('status','checked_cases','checked_joint_steps','elapsed_seconds')}))

if __name__ == '__main__':
    main()
