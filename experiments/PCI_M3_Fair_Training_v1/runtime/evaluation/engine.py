"""Complete-cohort evaluation; no training, GPU, proof or cloud operation.

The CLI is fail-closed and requires an explicit source-bound evaluation lock.
Outputs must be a new source-external subdirectory under a marker-bound owned
hot root; persistence/filesystem and staging checks belong to the controller.
Every scientific step samples a proposal from checked CPU Torch probabilities,
executes the native common checker and reconciles an independent scalar effect.
"""
from __future__ import annotations
import argparse, copy, gzip, json, os, sys, time
from pathlib import Path
import numpy as np
from .common import require, canonical, digest, sha, read_json, write_new_json, local, hash_string
from .actor import CheckedTorchActor, load_export, inverse_cdf_actions, sampling_probabilities
from .reference import scalar_world as scalar

METHODS = ('task', 'fixed', 'adaptive', 'semantic', 'consensus', 'directrisk')
PROFILES = {
    'nominal': dict(layout='twodoors', n_agents=4, horizon=256, quota_window=64, quota=3, capacity=2, rule_change=False),
    'quota_tight': dict(layout='twodoors', n_agents=4, horizon=256, quota_window=64, quota=1, capacity=2, rule_change=False),
    'capacity_tight': dict(layout='twodoors', n_agents=4, horizon=256, quota_window=64, quota=3, capacity=1, rule_change=False),
    'rule_shift': dict(layout='twodoors', n_agents=4, horizon=256, quota_window=64, quota=3, capacity=1, rule_change=True),
}
NATIVE_SHA = 'cc2ba5ad437b517f9aa3ab96119ce3fa3dba2c51098a7ff2d77ef6977f3e059c'
SCALAR_SHA = 'fad3c737516afae8623d527f121fcfc20856b559dd719c1c0a1c9ebb9ac55cef'
COUNT_KEYS = ('physical_agent_replacements', 'institutional_stage_agent_replacements', 'total_agent_replacements',
              'physical_bad_joint_steps', 'institutional_bad_joint_steps', 'quota_bad_joint_steps', 'capacity_bad_joint_steps',
              'proposed_compliant_joint_steps', 'deliveries', 'pickups',
              'wall_agent_attempts', 'collision_agent_attempts', 'raw_quota_violating_agent_decisions',
              'raw_capacity_excess', 'executed_compliant_joint_steps', 'moved_agent_decisions', 'nonadditive_stage_steps')


def checked_adapter(project_root):
    path = Path(project_root) / 'training/pci_risk_train/environment.py'
    require(sha(path) == NATIVE_SHA, 'IMMUTABLE_NATIVE_SOURCE')
    require(sha(Path(scalar.__file__)) == SCALAR_SHA, 'IMMUTABLE_SCALAR_SOURCE')
    return scalar.WorldAdapter(path)


def stage_counts(before, proposal, admitted, physical_actions, result):
    n = len(proposal)
    p = sum(a != b for a, b in zip(proposal, physical_actions))
    i = sum(a != b for a, b in zip(physical_actions, admitted))
    t = sum(a != b for a, b in zip(proposal, admitted))
    m = result['metrics']
    wall, collision = m['wall_attempts'], m['collision_attempts']
    quota, excess = m['quota_violations'], m['capacity_excess']
    require(t == m['interventions'], 'TOTAL_INTERVENTION_IDENTITY')
    require(m['proposed_compliant'] == int(not(wall or collision or quota or excess)), 'PROPOSAL_COMPONENT_IDENTITY')
    require(abs(p-i) <= t <= p+i <= 2*n, 'HAMMING_TRIANGLE')
    return dict(physical_agent_replacements=p, institutional_stage_agent_replacements=i,
                total_agent_replacements=t, physical_bad_joint_steps=int(wall > 0 or collision > 0),
                institutional_bad_joint_steps=int(quota > 0 or excess > 0),
                quota_bad_joint_steps=int(quota > 0), capacity_bad_joint_steps=int(excess > 0),
                proposed_compliant_joint_steps=m['proposed_compliant'], deliveries=m['delivered'],
                pickups=m['pickups'], wall_agent_attempts=wall, collision_agent_attempts=collision,
                raw_quota_violating_agent_decisions=quota, raw_capacity_excess=excess,
                executed_compliant_joint_steps=m['executed_compliant'], moved_agent_decisions=m['move_count'],
                nonadditive_stage_steps=int(p+i != t))


def physical_actions_native(env, proposal):
    candidate, _ = env.candidates(np.asarray(proposal, dtype=np.int64))
    physical, _ = env.physical_resolve(candidate)
    return env.actions_for_positions(physical)


def physical_actions_scalar(before, proposal):
    walls, _ = scalar.geometry(before['config']['layout'])
    raw = []
    for old, action in zip(before['pos'], proposal):
        dx, dy = scalar.MOVES[action]
        target = [old[0]+dx, old[1]+dy]
        raw.append(old[:] if tuple(target) in walls else target)
    positions, _ = scalar._physical(before['pos'], raw)
    return [scalar.MOVES.index((p[0]-o[0], p[1]-o[1])) for p, o in zip(positions, before['pos'])]


def validate_plan(protocol):
    require(protocol.get('methods') == list(METHODS) and protocol.get('profiles') == PROFILES, 'DECLARED_METHODS_AND_PROFILE_CONFIGS')
    training = protocol['training_cases']
    evaluation = protocol['evaluation_cases']
    require(type(training) is list and len(training) == 60, 'EXACT_SIXTY_TRAINING_CASES')
    require(type(evaluation) is list and len(evaluation) == 1200, 'EXACT_1200_EVALUATION_CASES')
    seeds = sorted({c['training_seed'] for c in training})
    require(len(seeds) == 10 and all(type(s) is int for s in seeds), 'EXACT_TEN_TRAINING_SEEDS')
    require(all(s not in range(2201, 2231) and s != 930000 for s in seeds), 'PROTECTED_SEED_RANGE')
    expected = {(m, s) for m in METHODS for s in seeds}
    require({(c['method'], c['training_seed']) for c in training} == expected, 'COMPLETE_TRAINING_CROSS')
    require(len({c['case_id'] for c in training}) == 60, 'UNIQUE_TRAINING_CASE_IDS')
    tm = {c['case_id']: c for c in training}
    keys = set()
    paired_streams = {}
    for c in evaluation:
        key = (c['method'], c['training_seed'], c['profile_id'], c['world_index'])
        require(key not in keys, 'DUPLICATE_EVALUATION_CELL')
        keys.add(key)
        require(c['method'] in METHODS and c['training_seed'] in seeds and c['profile_id'] in PROFILES and c['world_index'] in range(5), 'EVALUATION_CELL')
        require(c['horizon'] == 256 and c['training_case_id'] in tm, 'EVALUATION_BINDING')
        parent = tm[c['training_case_id']]
        require(parent['method'] == c['method'] and parent['training_seed'] == c['training_seed'], 'TRAIN_EVAL_IDENTITY')
        streamkey = (c['training_seed'], c['world_index'])
        stream = (c['world_seed'], c['policy_sampling_seed'])
        require(all(type(x) is int and x >= 0 for x in stream), 'NONNEGATIVE_STREAM_SEEDS')
        require(paired_streams.setdefault(streamkey, stream) == stream, 'EXOGENOUS_STREAM_PAIRING')
    require(keys == {(m,s,p,w) for m in METHODS for s in seeds for p in PROFILES for w in range(5)}, 'FULL_EVALUATION_CROSS')
    require(len({c['case_id'] for c in evaluation}) == 1200, 'UNIQUE_EVALUATION_CASE_IDS')
    return training, evaluation, seeds


def training_source_closure(project_root):
    names = ['vectorized_admission.py'] + ['pci_risk_train/'+n for n in
        ('__init__.py','environment.py','model.py','objectives.py','trainer.py','io.py')]
    return {n:sha(Path(project_root)/'training'/n) for n in sorted(names)}


def validate_closed_training_files(project_root, folder, config, done):
    require(done.get('schema') == 'PCI-RISK-TRAIN-COMPLETE-v1' and done.get('status') == 'COMPLETE' and done.get('run_role') == 'scientific', 'SCIENTIFIC_COMPLETE_TRAINING_RECEIPT')
    require(done['method'] == config['method'] and done['seed'] == config['seed'] and done['run_id'] == config['run_id'], 'TRAINING_RECEIPT_IDENTITY')
    require(done['config_sha256'] == digest(config), 'TRAINING_CANONICAL_CONFIG_IDENTITY')
    sources = training_source_closure(project_root)
    require(done['sources'] == sources, 'COMPLETE_TRAINING_SOURCE_CLOSURE')
    require(sha(folder/'CLOSED_FILES.json') == done['closed_manifest_sha256'], 'CLOSED_TRAINING_MANIFEST_BODY')
    manifest = read_json(folder/'CLOSED_FILES.json')
    require(manifest['schema'] == 'PCI-RISK-TRAIN-CLOSED-FILES-v1' and manifest['signature'] == done['signature'], 'CLOSED_TRAINING_MANIFEST_IDENTITY')
    files = manifest['files']
    required = {'config.json','provenance.json','checkpoint.pt','policy_0.pt','updates.csv','train_episodes.csv','LAST_COMMIT.json','final_export.npz'}
    require(required <= set(files) and {n for n in files if n.startswith('rollouts/')} == {f'rollouts/update_{i:06}.npz' for i in range(1,257)}, 'ALL_256_RETAINED_TRAINING_ROLLOUTS_REQUIRED')
    actual = {str(f.relative_to(folder)) for f in folder.rglob('*') if f.is_file()}
    require(actual == set(files)|{'TRAIN_COMPLETE.json','CLOSED_FILES.json'}, 'NO_MISSING_OR_UNDECLARED_TRAINING_FILES')
    for n,v in files.items():
        f=local(folder,n)
        require(f.is_file() and not f.is_symlink() and f.stat().st_size == v['bytes'] > 0 and sha(f) == v['sha256'], 'RETAINED_TRAINING_FILE:'+n)
    prov = read_json(folder/'provenance.json')
    require(prov['sources'] == sources and prov['signature'] == done['signature'] and prov['common_admission'] is True, 'TRAINING_PROVENANCE_CLOSURE')
    require(done['signature'] == digest({'config':config,'sources':sources,'versions':prov['system']['versions']}), 'TRAINING_SIGNATURE_INPUTS')
    require(done['episodes'] == 1024 and np.array_equal(np.asarray(done['phase_counts']),np.full((4,64),1024)), 'COMPLETE_TRAINING_EPISODE_PHASE_COUNTS')
    require(done['counters']['oracle_states'] == done['counters']['sampled_native_oracle_equal'] == 262144 and done['counters']['optimizer_steps'] > 0, 'TRAINING_ORACLE_AND_OPTIMIZER_COUNTS')
    return manifest


def training_work(config, done):
    return dict(env_steps=done['env_steps'],updates=done['updates'],episodes=done['episodes'],
        elapsed_seconds=done['elapsed_s'],counters=copy.deepcopy(done['counters']),
        phase_counts=copy.deepcopy(done['phase_counts']),configured_device=config['device'],torch_threads=config['torch_threads'],
        scope='Configured device and retained per-policy elapsed/optimizer/oracle counters. Sum of policy elapsed times is not cohort wall time; logical625 table rows are not625 native/scalar calls.')


def validate_training_cohort(project_root, training_root, training_cases):
    """Check ALL final models before constructing the first evaluation world."""
    import torch
    result = {}
    for c in training_cases:
        folder = local(training_root, c['case_id'])
        config = read_json(folder / 'config.json')
        done = read_json(folder / 'TRAIN_COMPLETE.json')
        require(config == c['config'], 'EXACT_FINAL_FROZEN_CONFIG')
        require(config['method'] == c['method'] and config['seed'] == c['training_seed'], 'FINAL_CONFIG_IDENTITY')
        require(config['total_steps'] == done['env_steps'] == 262144 and done['updates'] == 256, 'FINAL_FIXED_TRAINING_BUDGET')
        validate_closed_training_files(project_root,folder,config,done)
        checkpoint, export = folder/'checkpoint.pt', folder/'final_export.npz'
        require(sha(checkpoint) == done['checkpoint_sha256'], 'FINAL_CHECKPOINT_DIGEST')
        require(sha(export) == done['portable_export_sha256'], 'FINAL_EXPORT_DIGEST')
        hash_string(done['signature'])
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(ck['signature'] == done['signature'] and ck['step'] == 262144 and ck['update'] == 256, 'FINAL_CHECKPOINT_STATE')
        arrays = load_export(export)
        require(set(ck['model']) == set(arrays), 'CHECKPOINT_EXPORT_KEYS')
        for k, a in arrays.items():
            require(np.array_equal(a, ck['model'][k].detach().cpu().numpy()), 'CHECKPOINT_EXPORT_ARRAY:' + k)
        result[c['case_id']] = dict(training_case=copy.deepcopy(c), config_sha256=sha(folder/'config.json'),
            train_complete_sha256=sha(folder/'TRAIN_COMPLETE.json'), checkpoint_sha256=sha(checkpoint),
            portable_export_sha256=sha(export), signature=done['signature'], export_path=str(export),
            closed_training_manifest_sha256=done['closed_manifest_sha256'], training_work=training_work(config,done))
    require(len(result) == 60, 'COMPLETE_MODEL_GATE')
    return result


def episode_header(case, protocol_sha256, binding, fixture=False):
    return dict(schema='PCI-M2-EVALUATION-TRACE-HEADER-v1', case=copy.deepcopy(case),
        protocol_sha256=protocol_sha256, model_signature=binding['signature'],
        checkpoint_sha256=binding['checkpoint_sha256'], portable_export_sha256=binding['portable_export_sha256'],
        scientific=not fixture, policy_authority='CPU_TORCH_FLOAT32_WITH_PER_STATE_INDEPENDENT_NUMPY_FORWARD_CHECK',
        sampling='PCG64_FOUR_UNIFORMS_PER_STEP_INVERSE_CDF_AFTER_FLOAT64_RENORMALIZATION')


def empty_stratum(**labels):
    return dict(**labels,joint_steps=0,agent_decisions=0,counts=dict.fromkeys(COUNT_KEYS,0),task_reward_sum=0.0)


def add_stratum(stratum, counts, task_reward):
    stratum['joint_steps'] += 1
    stratum['agent_decisions'] += 4
    for k in COUNT_KEYS:
        stratum['counts'][k] += counts[k]
    stratum['task_reward_sum'] += task_reward


def run_episode(case, cfg, actor, adapter, output_root, protocol_sha256, binding, *, fixture=False):
    """One new world episode. Call only from the gated cohort or engineering fixture."""
    cfg = scalar.config(cfg)
    require(cfg['n_agents'] == 4 and case['horizon'] == cfg['horizon'], 'FOUR_AGENT_HORIZON_MATCH')
    out = Path(output_root)
    (out/'traces').mkdir(exist_ok=True)
    (out/'cases').mkdir(exist_ok=True)
    trace_name = f"traces/{case['case_id']}.jsonl.gz"
    trace = local(out, trace_name)
    partial = trace.with_suffix(trace.suffix+'.partial')
    require(not trace.exists() and not partial.exists(), 'NO_EVALUATION_CASE_REPLACEMENT')
    env = adapter.mod.Warehouse(adapter.mod.EnvConfig(**cfg), seed=case['world_seed'])
    policy_rng = np.random.default_rng(case['policy_sampling_seed'])
    header = episode_header(case, protocol_sha256, binding, fixture)
    chain = digest(header)
    counts = dict.fromkeys(COUNT_KEYS, 0)
    reward_sum = 0.0
    phases = [empty_stratum(phase=p) for p in range(cfg['quota_window'])]
    halves = [empty_stratum(half=h) for h in range(2)]
    first = adapter.snapshot(env)
    parity_max = {'logit_abs_error': 0.0, 'probability_abs_error': 0.0}
    with partial.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            stream.write((canonical({'header': header, 'chain': chain})+'\n').encode())
            for t in range(cfg['horizon']):
                before = adapter.snapshot(env)
                logits, probabilities, parity = actor.probabilities(env.features())
                rng_before = copy.deepcopy(policy_rng.bit_generator.state)
                uniforms = policy_rng.random(4)
                proposal = inverse_cdf_actions(probabilities, uniforms).tolist()
                rng_after = copy.deepcopy(policy_rng.bit_generator.state)
                physical = physical_actions_native(env, proposal)
                require(physical == physical_actions_scalar(before, proposal), 'PHYSICAL_NATIVE_SCALAR_PARITY')
                _, reward, cost, done, metrics = env.transition(proposal, enforce=True)
                effect = dict(before=before, proposal=proposal, admitted_actions=list(env.last['actions']),
                    after=adapter.snapshot(env), result=dict(reward=[float(x) for x in reward],
                        cost=[float(x) for x in cost], done=bool(done), metrics=metrics), record=copy.deepcopy(env.last))
                expected = scalar.scalar_effect(before, proposal)
                require(scalar.canonical(effect) == scalar.canonical(expected), 'NATIVE_SCALAR_FULL_EFFECT_PARITY')
                stage = stage_counts(before, proposal, effect['admitted_actions'], physical, effect['result'])
                payload = dict(case_id=case['case_id'], step=t, effect=effect,
                    physical_stage_actions=physical, stage_counts=stage,
                    policy=dict(logits=logits.tolist(), probabilities=probabilities.tolist(),
                        uniforms=uniforms.tolist(), rng_before=rng_before, rng_after=rng_after),
                    forward_parity=parity)
                next_chain = digest({'previous': chain, 'payload': payload})
                stream.write((canonical(dict(previous=chain, payload=payload, chain=next_chain))+'\n').encode())
                chain = next_chain
                for k, v in stage.items():
                    counts[k] += int(v)
                reward_sum += metrics['task_reward']
                add_stratum(phases[t % cfg['quota_window']],stage,metrics['task_reward'])
                add_stratum(halves[int(t >= cfg['horizon']//2)],stage,metrics['task_reward'])
                for k in parity_max:
                    parity_max[k] = max(parity_max[k], parity[k])
                require(bool(done) == (t+1 == cfg['horizon']), 'ONLY_TRUE_HORIZON_TERMINATION')
            final = adapter.snapshot(env)
            require(final['t'] == cfg['horizon'] and final['episode'] == first['episode'], 'COMPLETE_SINGLE_EPISODE')
            require(counts['deliveries'] == sum(final['delivered'])-sum(first['delivered']), 'DELIVERY_FRAME')
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(partial, trace)
    trace_sha = sha(trace)
    require(sha(trace) == trace_sha and trace.stat().st_size > 0, 'CLOSED_TRACE_RETENTION')
    summary = dict(schema='PCI-M2-EVALUATION-CASE-v1', status='PASS', case=copy.deepcopy(case),
        scientific=not fixture, config=cfg, trace_path=trace_name, trace_sha256=trace_sha,
        trace_bytes=trace.stat().st_size, trace_header_sha256=digest(header), final_chain=chain,
        initial_world_sha256=scalar.digest(first), final_world_sha256=scalar.digest(final),
        joint_steps=cfg['horizon'], agent_decisions=4*cfg['horizon'], counts=counts,
        task_reward_sum=reward_sum, phase_summaries=phases, half_summaries=halves,
        forward_parity_max=parity_max, model_binding=copy.deepcopy(binding))
    write_new_json(local(out, f"cases/{case['case_id']}.json"), summary)
    return summary


def source_closure(project_root):
    root = Path(project_root).resolve()
    names = ['evaluation/'+str(p.relative_to(root/'evaluation')) for p in sorted((root/'evaluation').rglob('*.py'))
             if '__pycache__' not in p.parts]
    names += ['training/'+n for n in training_source_closure(root)]
    return {n: sha(root/n) for n in names}


def checked_lock(project_root, lock_path):
    lock = read_json(lock_path)
    require(lock.get('schema') == 'PCI-M2-EVALUATION-EXECUTION-LOCK-v1', 'EVALUATION_LOCK_SCHEMA')
    require(lock.get('scientific_launch_authorized') is True, 'SCIENTIFIC_LAUNCH_NOT_AUTHORIZED')
    require(lock.get('source_sha256') == source_closure(project_root), 'FROZEN_EVALUATION_SOURCE_CLOSURE')
    protocol_path = local(project_root, lock['protocol_path'])
    require(sha(protocol_path) == lock['protocol_sha256'], 'FROZEN_PROTOCOL_IDENTITY')
    protocol = read_json(protocol_path)
    require(protocol.get('schema') == 'PCI-M2-MATCHED-RISK-TRAINING-PROTOCOL-v1' and protocol.get('scientific_design_frozen') is True, 'FROZEN_SCIENTIFIC_PROTOCOL_REQUIRED')
    validate_plan(protocol)
    return lock, protocol


def checked_hot_output(project_root, output, lock, protocol):
    root=Path(project_root).resolve()
    require(Path(output).is_absolute(),'ABSOLUTE_EVALUATION_OUTPUT_REQUIRED')
    require(type(lock.get('owned_hot_output_root')) is str and Path(lock['owned_hot_output_root']).is_absolute(),'ABSOLUTE_OWNED_HOT_ROOT_REQUIRED')
    hot=Path(lock['owned_hot_output_root']).resolve()
    out=Path(output).resolve()
    require(str(hot) not in ('/','/home','/workspace','/mnt','/tmp','/root'),'NARROW_OWNED_HOT_ROOT_REQUIRED')
    require(not hot.is_relative_to(root) and not root.is_relative_to(hot),'HOT_ROOT_MUST_NOT_OVERLAP_SOURCE')
    require(out!=hot and out.is_relative_to(hot) and not out.is_relative_to(root),'NEW_EVALUATION_SUBDIRECTORY_OF_OWNED_HOT_ROOT')
    marker=hot/'HOT_ROOT_OWNERSHIP.json'
    require(marker.is_file() and not marker.is_symlink() and sha(marker)==lock.get('owned_hot_output_marker_sha256'),'OWNED_HOT_ROOT_MARKER_IDENTITY')
    observed=read_json(marker)
    require(observed.get('schema')=='PCI-M2-OWNED-HOT-ROOT-v1' and observed.get('created_by_controller') is True,'OWNED_HOT_ROOT_MARKER_SCHEMA')
    require(observed.get('study_id')==protocol['study_id'] and observed.get('protocol_sha256')==lock['protocol_sha256'],'HOT_ROOT_STUDY_PROTOCOL_BINDING')
    require(observed.get('project_root')==str(root) and observed.get('owned_hot_output_root')==str(hot),'HOT_ROOT_SOURCE_PATH_BINDING')
    require(not out.exists(),'NEW_EVALUATION_COHORT_DIRECTORY_REQUIRED')
    return out


def evaluate(project_root, training_root, lock_path, output, *, operations_root, deadline_monotonic):
    root=Path(project_root).resolve()
    lock, protocol = checked_lock(root, lock_path)
    output=checked_hot_output(root,output,lock,protocol)
    training, cases, _ = validate_plan(protocol)
    bindings = validate_training_cohort(root, training_root, training)
    adapter = checked_adapter(root)
    # All sixty closed models have passed the checkpoint/export gate at this point.
    output.mkdir(mode=0o700)
    write_new_json(output/'PROVENANCE.json', dict(schema='PCI-M2-EVALUATION-PROVENANCE-v1',
        protocol_sha256=lock['protocol_sha256'], execution_lock_sha256=sha(lock_path),
        model_bindings=bindings, source_sha256=source_closure(root), policy_sampling='PAIRED_FIXED_FOUR_UNIFORMS',
        certified_transitions=0, GPU_calls=0, scope='New CPU evaluation of final policies after common native admission; not cryptographic certification.'))
    start = time.perf_counter()
    try:
        from .parallel import run_phase
        deadline = min(deadline_monotonic, time.monotonic()+lock['wall_ceiling_seconds'])
        phase = run_phase(root,output,cases,bindings,lock['protocol_sha256'],'native',
                          operations_root,deadline,expected_source_sha256=lock['source_sha256'])
        summaries = phase['summaries']
        require(len(summaries) == 1200, 'ALL_EVALUATION_CASES_REQUIRED')
        require(source_closure(root) == lock['source_sha256'], 'SOURCE_CHANGED_DURING_EVALUATION')
        for b in bindings.values():
            require(sha(b['export_path']) == b['portable_export_sha256'], 'MODEL_CHANGED_DURING_EVALUATION')
        files = {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(output.rglob('*')) if p.is_file()}
        manifest = dict(schema='PCI-M2-EVALUATION-CLOSED-MANIFEST-v1', status='PASS', scientific=True,
            complete_cases=1200, joint_steps=307200, agent_decisions=1228800,
            protocol_sha256=lock['protocol_sha256'], execution_lock_sha256=sha(lock_path),
            elapsed_seconds=time.perf_counter()-start, files=files,
            scope='Closed primary evaluation records; independent retained audit and complete-family analysis are separate requirements.')
        write_new_json(output/'EVALUATION_COMPLETE.json', manifest)
        return manifest
    except BaseException as exc:
        statuses = getattr(exc, 'receipt', {}).get('case_statuses', [])
        if not statuses:
            from .parallel import phase_ledger
            statuses = phase_ledger(output,operations_root,cases,'native')
        completed = [r['case_id'] for r in statuses if r.get('closed_case')]
        failed = [r['case_id'] for r in statuses if r['status'] == 'FAILED_CURRENT_CASE']
        write_new_json(output/'EVALUATION_FAILED.json', dict(schema='PCI-M2-EVALUATION-FAILURE-v1',
            status='FAIL_NO_SCIENTIFIC_ANALYSIS', completed_cases=len(completed),
            error=repr(exc), failed_case_id=failed[0] if len(failed)==1 else None,
            failed_case_ids=failed, case_statuses=statuses,
            failed_cases_not_replaced=True, elapsed_seconds=time.perf_counter()-start))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--execution-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--operations-root', type=Path, required=True)
    p.add_argument('--deadline-monotonic', type=float, required=True)
    a = p.parse_args()
    result = evaluate(a.project_root, a.training_root, a.execution_lock, a.output,
                      operations_root=a.operations_root,deadline_monotonic=a.deadline_monotonic)
    print(json.dumps({k: result[k] for k in ('status','complete_cases','joint_steps','elapsed_seconds')}))

if __name__ == '__main__':
    main()
