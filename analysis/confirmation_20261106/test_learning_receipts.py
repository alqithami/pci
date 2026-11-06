"""Artificial receipt controls only; never reads live scientific outcomes."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import learning_receipts as lr

ROOT = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/workspace/scratch/5bbfb44aa122/pci_work_20261106')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE / 'protocol'))
import build_study as study
import frontier
import native_profiles


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root, domain):
    snapshot = lr.Snapshot()
    sources, _ = lr._protocol(SOURCE, snapshot)
    cohort = root / 'cohort'
    inputs = root / 'inputs'
    evaluation = cohort / 'evaluation'
    put(cohort / 'SYNTHETIC_TEST_ONLY.json', {'scope': 'ARTIFICIAL SOFTWARE CONTROL; NOT SCIENTIFIC RESULTS'})
    protocol = study.protocol()
    ph = lr.digest(protocol)
    put(inputs / 'cases/PROTOCOL_DRAFT.json', protocol)
    devseeds = study.DEVELOPMENT_SEEDS if domain == 'warehouse' else study.RWARE_DEVELOPMENT_SEEDS
    seeds = study.CONFIRMATION_SEEDS if domain == 'warehouse' else study.RWARE_CONFIRMATION_SEEDS
    rows = [{'candidate_id': r['candidate_id'], 'family': r['family'], 'training_seed': seed,
             'domain': domain, 'n_agents': 4, 'joint_steps': 256,
             'institutional_replacements': 0, 'deliveries': 0,
             'cohort_complete': True, 'evaluation_complete': True}
            for seed in devseeds for r in study.candidates()]
    selection = frontier.select(rows, source_sha256=lr.digest(rows), study_domain=domain)
    selection_path = root / 'DEVELOPMENT_SELECTION.json'
    put(selection_path, selection)
    native = native_profiles.build()['profiles_by_n']['4'] if domain == 'rware' else None
    selected = selection['selected_candidate_ids']
    choices = {r['candidate_id']: r for r in study.candidates()}
    configs, descriptors = {}, []
    for seed in seeds:
        for candidate in selected:
            value = study.train_config(choices[candidate], seed, seeds.index(seed), role='confirmatory',
                steps=524288, lock_hash=ph, stage='confirmation', domain=domain,
                domain_profiles=native['profiles'] if native else None,
                domain_slots=native.get('env_profile_ids') if native else None)
            run = value['run_id']
            path = inputs / 'cases' / (run + '.json')
            put(path, value)
            configs[run] = value
            descriptors.append({'case_id': run, 'candidate_id': candidate, 'family': value['method'],
                'domain': domain, 'n_agents': 4, 'training_seed': seed, 'path': path.name,
                'sha256': sha(path), 'run_role': 'confirmatory', 'total_steps': 524288,
                'risk_estimator': value['risk_estimator'], 'actual_auxiliary_coef': value['auxiliary_coef']})
    ledger = {'schema': 'PCI-COMPLETE-CASE-MANIFEST-v1', 'stage': 'confirmation', 'study_domain': domain,
        'case_count': 160, 'protocol_canonical_sha256': ph, 'cases': descriptors}
    ledger_path = inputs / 'cases/CASE_MANIFEST.json'
    put(ledger_path, ledger)
    cfg = {'schema': 'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1', 'role': 'scientific', 'stage': 'confirmation',
        'study_domain': domain, 'device': 'cuda', 'worlds_per_profile': 20,
        'out': '/synthetic/' + domain + '/cohort', 'deadline_unix': 2,
        'source_design_lock': '/synthetic/inputs/SOURCE_DESIGN_LOCK.json',
        'launch_lock': '/synthetic/inputs/LAUNCH_LOCK.json',
        'readiness_receipt': '/synthetic/inputs/CUDA_READINESS.json',
        'protocol_file': '/synthetic/inputs/cases/PROTOCOL_DRAFT.json',
        'expected_cells': [{'domain': domain, 'n_agents': 4, 'candidate_ids': selected, 'training_seeds': seeds}]}
    put(inputs / 'CONTROLLER_CONFIG.json', cfg)
    design = {'schema': 'PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1', 'scientific_launch_authorized': True,
        'engineering_complete': True, 'source_sha256': sources, 'source_map_sha256': lr.PINNED_SOURCE,
        'protocol_canonical_sha256': ph, 'development_selection_sha256': sha(selection_path),
        'authorized_stage': 'confirmation'}
    if native:
        design['native_rware_profiles_by_n'] = {'4': native}
    put(inputs / 'SOURCE_DESIGN_LOCK.json', design)
    ready = {'schema': 'PCI-FAIR-CUDA-READINESS-v1', 'status': 'PASS', 'all_required_methods_pass': True,
        'source_map_sha256': lr.PINNED_SOURCE, 'device': 'cuda', 'host_boot_id': 'SYNTHETIC', 'gpu_uuid': 'SYNTHETIC'}
    put(inputs / 'CUDA_READINESS.json', ready)
    comparisons = {'controller_config_sha256': lr.digest(cfg), 'case_manifest_sha256': sha(ledger_path),
        'source_design_lock_sha256': sha(inputs / 'SOURCE_DESIGN_LOCK.json'),
        'readiness_receipt_sha256': sha(inputs / 'CUDA_READINESS.json'), 'source_map_sha256': lr.PINNED_SOURCE,
        'out': cfg['out'], 'deadline_unix': cfg['deadline_unix']}
    launch = {'schema': 'PCI-FAIR-SCIENTIFIC-LAUNCH-v1', 'scientific_launch_authorized': True,
        'host_boot_id': 'SYNTHETIC', 'gpu_uuid': 'SYNTHETIC', **comparisons}
    put(inputs / 'LAUNCH_LOCK.json', launch)
    external = {cfg[key]: sha(inputs / name) for key, name in (
        ('source_design_lock', 'SOURCE_DESIGN_LOCK.json'), ('launch_lock', 'LAUNCH_LOCK.json'),
        ('readiness_receipt', 'CUDA_READINESS.json'), ('protocol_file', 'cases/PROTOCOL_DRAFT.json'))}
    ex = {'controller_configuration': cfg, 'controller_configuration_file_sha256': sha(inputs / 'CONTROLLER_CONFIG.json'),
        'source_sha256': sources, 'case_manifest_sha256': sha(ledger_path),
        'authority': {**comparisons, 'host_boot_id': 'SYNTHETIC', 'gpu_uuid': 'SYNTHETIC', 'external_record_sha256': external}}
    put(cohort / 'EXECUTION_LOCK.json', ex)
    policies, cases = [], []
    for row in descriptors:
        run = row['case_id']
        value = configs[run]
        policy = {key: row[key] for key in ('candidate_id', 'family', 'training_seed', 'domain', 'n_agents')}
        policy.update(directory=run, run_id=run, training_signature=lr.digest(['signature', run]),
                      portable_export_sha256=lr.digest(['export', run]))
        policies.append(policy)
        for profile, ecfg in sorted(value['profiles'].items()):
            for world in range(20):
                ws = 3400000000 + (100000000 if domain == 'rware' else 0) + 4000000 + 1000 * seeds.index(row['training_seed']) + world
                cases.append({'case_id': f'{run}__{profile}__w{world:02d}', 'policy_directory': run,
                    'candidate_id': row['candidate_id'], 'family': row['family'], 'method': row['family'],
                    'training_seed': row['training_seed'], 'domain': domain, 'n_agents': 4,
                    'profile_id': profile, 'environment_config': ecfg, 'world_index': world,
                    'world_seed': ws, 'policy_seed': ws + 50000000})
    manifest = {'schema': 'PCI-FAIR-EVALUATION-MANIFEST-v1', 'stage': 'confirmation', 'run_role': 'confirmatory',
        'training_case_manifest': ledger, 'training_case_manifest_sha256': lr.digest(ledger),
        'protocol_sha256': ph, 'worlds_per_profile': 20, 'training_seed_clusters': seeds,
        'policies': policies, 'cases': cases, 'complete_policy_count': 160, 'complete_episode_count': len(cases),
        'source_sha256': {key.removeprefix('runtime/'): value for key, value in sources.items()
            if key.startswith(('runtime/training/', 'runtime/evaluation_fair/'))}}
    put(cohort / 'EVALUATION_MANIFEST.json', manifest)
    put(evaluation / 'MANIFEST.json', manifest)
    mh = lr.digest(manifest)
    receipts, intents, reports = {}, {}, []
    profile_rows, seed_rows = [], []
    for policy in policies:
        identity = {key: policy[key] for key in ('candidate_id', 'family', 'training_seed', 'domain', 'n_agents')}
        profiles = sorted(configs[policy['run_id']]['profiles'])
        shield = policy['family'] == 'shieldaware'
        unit = {'physical_replacements': 0 if shield else 2, 'institutional_replacements': 0 if shield else 1,
            'total_replacements': 0 if shield else 3, 'deliveries': 2, 'pickups': 2, 'joint_steps': 256}
        for profile in profiles:
            profile_rows.append({**identity, 'profile_id': profile, **{key: value * 20 for key, value in unit.items()}})
        counts = {key: value * 20 * len(profiles) for key, value in unit.items()}
        steps = counts['joint_steps']
        seed_rows.append({**identity, **counts, 'cohort_complete': True, 'evaluation_complete': True,
            'profile_weights': {profile: 1 / len(profiles) for profile in profiles},
            'institutional_per_1000_agent_decisions': 1000 * counts['institutional_replacements'] / (4 * steps),
            'deliveries_per_1000_joint_steps': 1000 * counts['deliveries'] / steps,
            'normalized_cost': counts['institutional_replacements'] / (4 * steps),
            'normalized_utility': counts['deliveries'] / (4 * steps)})
    policy_by_run = {row['run_id']: row for row in policies}
    for case in cases:
        key = case['case_id']
        policy = policy_by_run[case['policy_directory']]
        shield = case['family'] == 'shieldaware'
        totals = {'physical_replacements': 0 if shield else 2, 'institutional_replacements': 0 if shield else 1,
            'total_replacements': 0 if shield else 3, 'deliveries': 2, 'pickups': 2, 'joint_steps': 256}
        seal = {'sha256': lr.digest(['trace', key]), 'publication_intent_sha256': lr.digest(['intent', key]), 'records': 258}
        body = {'schema': 'PCI-FAIR-EVALUATION-CASE-v1', 'status': 'COMPLETE', 'case': case,
            'manifest_sha256': mh, 'training_signature': policy['training_signature'],
            'portable_export_sha256': policy['portable_export_sha256'], 'trace_seal': seal, 'totals': totals,
            'deployment_policy_requested_rows': 625 * 256 if shield else 0,
            'deployment_policy_callback_calls': 256 if shield else 0}
        path = evaluation / 'cases' / (key + '.json')
        put(path, body)
        receipts[key] = sha(path)
        intents[key] = seal['publication_intent_sha256']
        reports.append({'case_id': key, 'status': 'PASS', 'trace_sha256': seal['sha256'], 'joint_steps': 256, 'agent_decisions': 1024})
    analysis = {'schema': 'PCI-FAIR-EVALUATION-AGGREGATE-v1', 'status': 'PASS_COMPLETE_COHORT', 'stage': 'confirmation',
        'policy_count': 160, 'episode_count': len(cases), 'equal_profile_seed_rows': seed_rows, 'profile_seed_rows': profile_rows}
    put(evaluation / 'ANALYSIS.json', analysis)
    complete = {'schema': 'PCI-FAIR-EVALUATION-COMPLETE-v1', 'status': 'COMPLETE', 'manifest_sha256': mh,
        'analysis_sha256': sha(evaluation / 'ANALYSIS.json'), 'all_policy_count': 160, 'all_episode_count': len(cases),
        'joint_steps': len(cases) * 256, 'agent_decisions': len(cases) * 1024, 'case_receipts': receipts, 'trace_intent_receipts': intents}
    put(evaluation / 'EVALUATION_COMPLETE.json', complete)
    audit = {'schema': 'PCI-FAIR-RETAINED-RECORD-AUDIT-v1', 'status': 'PASS', 'manifest_sha256': mh,
        'policy_count': 160, 'episode_count': len(cases), 'joint_steps': len(cases) * 256, 'agent_decisions': len(cases) * 1024,
        'independent_numpy_actor': True, 'independent_scalar_stage_checks': True, 'recorded_action_native_replay': True,
        'replacement_policy_rollouts': 0, 'case_reports': reports}
    put(evaluation / 'RETAINED_RECORD_AUDIT.json', audit)
    final = {'schema': 'PCI-FAIR-COHORT-COMPLETE-v1', 'status': 'COMPLETE_AUDITED_COHORT', 'stage': 'confirmation',
        'study_domain': domain, 'trained_policies': 160, 'training_joint_steps': 83886080,
        'all_owned_worker_processes_closed': True, 'execution_lock_sha256': sha(cohort / 'EXECUTION_LOCK.json'),
        'source_map_sha256': lr.PINNED_SOURCE, 'case_manifest_sha256': sha(ledger_path),
        'absolute_deadline_unix': 2, 'completed_unix': 1, 'analysis_sha256': sha(evaluation / 'ANALYSIS.json'),
        'retained_record_audit_sha256': sha(evaluation / 'RETAINED_RECORD_AUDIT.json'),
        'evaluated_episodes': len(cases), 'evaluation_joint_steps': len(cases) * 256, 'evaluation_agent_decisions': len(cases) * 1024}
    put(cohort / 'STUDY_COMPLETE.json', final)
    return cohort, selection_path, ledger_path


def refuse(call, message):
    try:
        call()
    except (ValueError, FileNotFoundError) as error:
        assert message in str(error), (message, str(error))
        return str(error)
    raise AssertionError('Expected refusal: ' + message)


def main():
    global SOURCE
    controls = []
    with tempfile.TemporaryDirectory(prefix='pci_confirmation_adapter_synthetic_') as tmp:
        root = Path(tmp)
        # Reference qualification evidence is absent from the recovered source.
        # Exercise RWARE with explicit artificial evidence in a temporary source
        # clone; its 43 executable bytes remain identical to the pinned map.
        original_source = SOURCE
        sources, _ = lr._protocol(original_source, lr.Snapshot())
        SOURCE = root / 'synthetic_source'
        for name in sources:
            target = SOURCE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_source / name, target)
        native_profiles.ROOT = SOURCE
        qualification_root = SOURCE / 'validation/environments'
        put(qualification_root / 'REFERENCE_PROFILE_QUALIFICATION_PLAN.json',
            {'schema': 'SYNTHETIC_REFERENCE_CONTROL_ONLY', 'scientific_evidence': False})
        qualification = {'schema': 'PCI-RWARE-REFERENCE-QUALIFICATION-v1',
            'status': 'COMPLETE_REFERENCE_SCREEN_NOT_SCIENTIFIC_APPROVAL',
            'plan_sha256': sha(qualification_root / 'REFERENCE_PROFILE_QUALIFICATION_PLAN.json'),
            'native_reward_only': True, 'no_M3_learner_or_holdout_data_used': True,
            'sources': {'runtime/training/pci_risk_train/rware_adapter.py': sources['runtime/training/pci_risk_train/rware_adapter.py']},
            'cells': [{'profile_id': 'T_q1_w64_c1', 'layout': 'small', 'n_agents': n,
                       'qualification_pass': True, 'checks': {'synthetic_control': True}} for n in (4, 8, 12)],
            'rows': [], 'synthetic_only': True}
        put(qualification_root / 'REFERENCE_PROFILE_QUALIFICATION.json', qualification)
        for domain in ('warehouse', 'rware'):
            cohort, selection, ledger = make_fixture(root / domain, domain)
            invoke = lambda: lr.load_learning_receipts(cohort, SOURCE, selection, ledger, _test_only_synthetic=True)
            data = invoke()
            assert len(data['seed_rows']) == 160 and len(data['profile_rows']) == (640 if domain == 'warehouse' else 160)
            assert sum(row['joint_steps'] for row in data['seed_rows']) == (3276800 if domain == 'warehouse' else 819200)
            controls.append({'control': domain + ' complete synthetic source/receipt/config/RNG/count closure', 'status': 'PASS'})
            error = refuse(lambda: lr.load_learning_receipts(cohort, SOURCE, selection, ledger), 'SYNTHETIC_COHORT_REFUSED')
            controls.append({'control': domain + ' synthetic production refusal', 'status': 'PASS_REFUSED', 'error': error})
            if domain == 'rware':
                continue
            def mutate(path, change, label, error):
                original = path.read_bytes()
                try:
                    put(path, change(json.loads(original)))
                    controls.append({'control': label, 'status': 'PASS_REFUSED', 'error': refuse(invoke, error)})
                finally:
                    path.write_bytes(original)
            mutate(cohort / 'STUDY_COMPLETE.json', lambda v: dict(v, stage='development'), 'development cannot become confirmation', 'COMPLETE_CONFIRMATION_REQUIRED')
            mutate(cohort / 'STUDY_COMPLETE.json', lambda v: dict(v, schema='WRONG_PRODUCER_SCHEMA'), 'wrong completion producer schema', 'PRODUCER_SCHEMA:STUDY_COMPLETE.json')
            mutate(ledger.parent.parent / 'SOURCE_DESIGN_LOCK.json', lambda v: dict(v, development_selection_sha256='0' * 64), 'selection must be bound before confirmation', 'PROSPECTIVELY_BOUND_DEVELOPMENT_SELECTION_REQUIRED')
            mutate(selection, lambda v: dict(v, selected_candidate_ids=list(reversed(v['selected_candidate_ids']))), 'selection receipt tampering', 'PROSPECTIVELY_BOUND_DEVELOPMENT_SELECTION_REQUIRED')
            config_path = ledger.parent / json.loads(ledger.read_text())['cases'][0]['path']
            mutate(config_path, lambda v: dict(v, total_steps=262144), 'training budget tampering', 'TRAINING_CONFIG_HASH')
            audit = cohort / 'evaluation/RETAINED_RECORD_AUDIT.json'
            original = audit.read_bytes()
            audit.unlink()
            try:
                controls.append({'control': 'missing retained audit', 'status': 'PASS_REFUSED', 'error': refuse(invoke, 'RETAINED_RECORD_AUDIT.json')})
            finally:
                audit.write_bytes(original)
            missing_case = next((cohort / 'evaluation/cases').glob('*.json'))
            missing_body = missing_case.read_bytes()
            missing_case.unlink()
            try:
                controls.append({'control': 'missing prescribed evaluation case file', 'status': 'PASS_REFUSED',
                    'error': refuse(invoke, 'CASE_FILE_MEMBERSHIP')})
            finally:
                missing_case.write_bytes(missing_body)
            # Rebind every affected analysis hash so duplicate cluster rows
            # reach the semantic complete-Cartesian-product check.
            analysis_path = cohort / 'evaluation/ANALYSIS.json'
            completion_path = cohort / 'evaluation/EVALUATION_COMPLETE.json'
            final_path = cohort / 'STUDY_COMPLETE.json'
            originals = {path: path.read_bytes() for path in (analysis_path, completion_path, final_path)}
            try:
                analysis = json.loads(originals[analysis_path])
                analysis['equal_profile_seed_rows'].append(copy.deepcopy(analysis['equal_profile_seed_rows'][0]))
                put(analysis_path, analysis)
                for path in (completion_path, final_path):
                    value = json.loads(originals[path])
                    value['analysis_sha256'] = sha(analysis_path)
                    put(path, value)
                controls.append({'control': 'duplicate candidate-training-seed aggregate with rebound hashes',
                    'status': 'PASS_REFUSED', 'error': refuse(invoke, 'INCOMPLETE_OR_DUPLICATE:ANALYSIS_SEED_ROWS')})
            finally:
                for path, body in originals.items():
                    path.write_bytes(body)
            path = next((cohort / 'evaluation/cases').glob('*shieldaware*.json'))
            completion_path = cohort / 'evaluation/EVALUATION_COMPLETE.json'
            original, original_completion = path.read_bytes(), completion_path.read_bytes()
            try:
                value = json.loads(original)
                value['totals'].update(physical_replacements=1, total_replacements=1)
                put(path, value)
                completion = json.loads(original_completion)
                completion['case_receipts'][value['case']['case_id']] = sha(path)
                put(completion_path, completion)
                controls.append({'control': 'rebound impossible shield summary', 'status': 'PASS_REFUSED', 'error': refuse(invoke, 'SHIELD_REPLACEMENT_INVARIANT')})
            finally:
                path.write_bytes(original)
                completion_path.write_bytes(original_completion)
    result = {'schema': 'PCI-CONFIRMATION-RECEIPT-ADAPTER-TESTS-v1', 'status': 'PASS_SYNTHETIC_ADAPTER_CONTROLS',
        'synthetic_only': True, 'scientific_results_read': False, 'controls': controls,
        'total_checks': len(controls), 'source_map_sha256': lr.PINNED_SOURCE,
        'adapter_sha256': sha(ROOT / 'learning_receipts.py'), 'test_source_sha256': sha(Path(__file__))}
    put(ROOT / 'ADAPTER_TEST_RESULTS.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
