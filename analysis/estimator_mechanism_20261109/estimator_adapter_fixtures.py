"""Artificial receipt controls only; never reads live scientific outcomes."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import estimator_receipts as lr

ROOT = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/workspace/scratch/5bbfb44aa122/pci_finalization_20261109/confirmation_export/source')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE / 'protocol'))
import build_study as study
import frontier


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_unit(candidate_id, seed):
    # Deliberately unequal by mode, base configuration and seed. This prevents
    # a zero-contrast fixture from concealing a mode/seed mapping error.
    mode = candidate_id.rsplit('_', 1)[1]
    offset = {'exact': 0, 'score': 1, 'conditional': 2}[mode]
    index = seed - 64001
    high = candidate_id.startswith('direct_02_')
    return {'physical_replacements': 2,
            'institutional_replacements': 1 + int(high) + offset + index % 2,
            'total_replacements': 3 + int(high) + offset + index % 2,
            'deliveries': 8 + int(high) - offset + index % 3,
            'pickups': 12, 'joint_steps': 256}


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
    seeds = study.ESTIMATOR_LEARNING_SEEDS
    rows = [{'candidate_id': r['candidate_id'], 'family': r['family'], 'training_seed': seed,
             'domain': domain, 'n_agents': 4, 'joint_steps': 256,
             'institutional_replacements': int(r['candidate_id'] == 'direct_02'), 'deliveries': (2 if r['candidate_id'] == 'direct_02' else 1 if r['candidate_id'] == 'direct_11' else 0),
             'cohort_complete': True, 'evaluation_complete': True}
            for seed in devseeds for r in study.candidates()]
    selection = frontier.select(rows, source_sha256=lr.digest(rows), study_domain=domain)
    selection_path = root / 'DEVELOPMENT_SELECTION.json'
    put(selection_path, selection)
    native = None
    assert selection['estimator_ablation_candidate_ids'] == ['direct_02', 'direct_11']
    base_choices = {r['candidate_id']: r for r in study.candidates()}
    choices = {base+'_'+mode:dict(base_choices[base], candidate_id=base+'_'+mode, risk_estimator=mode) for base in ['direct_02','direct_11'] for mode in ['exact','score','conditional']}
    selected = list(choices)
    configs, descriptors = {}, []
    for seed in seeds:
        for candidate in selected:
            value = study.train_config(choices[candidate], seed, seeds.index(seed), role='confirmatory',
                steps=524288, lock_hash=ph, stage='estimator_ablation', domain=domain,
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
    ledger = {'schema': 'PCI-COMPLETE-CASE-MANIFEST-v1', 'stage': 'estimator_ablation', 'study_domain': domain,
        'case_count': 60, 'protocol_canonical_sha256': ph, 'cases': descriptors}
    ledger_path = inputs / 'cases/CASE_MANIFEST.json'
    put(ledger_path, ledger)
    cfg = {'schema': 'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1', 'role': 'scientific', 'stage': 'estimator_ablation',
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
        'stage': 'estimator_ablation', 'study_domain': 'warehouse', 'training_policies': 60, 'training_seeds': seeds,
        'estimator_analysis_recipe_sha256': sha(ROOT/'ESTIMATOR_ANALYSIS_RECIPE.json'),
        'estimator_analysis_implementation_lock_sha256': sha(ROOT/'ESTIMATOR_ANALYSIS_IMPLEMENTATION_LOCK.json')}
    if native:
        design['native_rware_profiles_by_n'] = {'4': native}
    put(inputs / 'SOURCE_DESIGN_LOCK.json', design)
    ready = {'schema': 'PCI-FAIR-CUDA-READINESS-v1', 'status': 'PASS', 'all_required_methods_pass': True,
        'estimator_modes_pass': {'exact': True, 'score': True, 'conditional': True},
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
                ws = 3804000000 + 1000 * seeds.index(row['training_seed']) + world
                cases.append({'case_id': f'{run}__{profile}__w{world:02d}', 'policy_directory': run,
                    'candidate_id': row['candidate_id'], 'family': row['family'], 'method': row['family'],
                    'training_seed': row['training_seed'], 'domain': domain, 'n_agents': 4,
                    'profile_id': profile, 'environment_config': ecfg, 'world_index': world,
                    'world_seed': ws, 'policy_seed': ws + 50000000})
    manifest = {'schema': 'PCI-FAIR-EVALUATION-MANIFEST-v1', 'stage': 'estimator_ablation', 'run_role': 'confirmatory',
        'training_case_manifest': ledger, 'training_case_manifest_sha256': lr.digest(ledger),
        'protocol_sha256': ph, 'worlds_per_profile': 20, 'training_seed_clusters': seeds,
        'policies': policies, 'cases': cases, 'complete_policy_count': 60, 'complete_episode_count': len(cases),
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
        unit = synthetic_unit(policy['candidate_id'], policy['training_seed'])
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
        totals = synthetic_unit(case['candidate_id'], case['training_seed'])
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
    analysis = {'schema': 'PCI-FAIR-EVALUATION-AGGREGATE-v1', 'status': 'PASS_COMPLETE_COHORT', 'stage': 'estimator_ablation',
        'policy_count': 60, 'episode_count': len(cases), 'equal_profile_seed_rows': seed_rows, 'profile_seed_rows': profile_rows}
    put(evaluation / 'ANALYSIS.json', analysis)
    complete = {'schema': 'PCI-FAIR-EVALUATION-COMPLETE-v1', 'status': 'COMPLETE', 'manifest_sha256': mh,
        'analysis_sha256': sha(evaluation / 'ANALYSIS.json'), 'all_policy_count': 60, 'all_episode_count': len(cases),
        'joint_steps': len(cases) * 256, 'agent_decisions': len(cases) * 1024, 'case_receipts': receipts, 'trace_intent_receipts': intents}
    put(evaluation / 'EVALUATION_COMPLETE.json', complete)
    audit = {'schema': 'PCI-FAIR-RETAINED-RECORD-AUDIT-v1', 'status': 'PASS', 'manifest_sha256': mh,
        'policy_count': 60, 'episode_count': len(cases), 'joint_steps': len(cases) * 256, 'agent_decisions': len(cases) * 1024,
        'independent_numpy_actor': True, 'independent_scalar_stage_checks': True, 'recorded_action_native_replay': True,
        'replacement_policy_rollouts': 0, 'case_reports': reports}
    put(evaluation / 'RETAINED_RECORD_AUDIT.json', audit)
    final = {'schema': 'PCI-FAIR-COHORT-COMPLETE-v1', 'status': 'COMPLETE_AUDITED_COHORT', 'stage': 'estimator_ablation',
        'study_domain': domain, 'trained_policies': 60, 'training_joint_steps': 31457280,
        'all_owned_worker_processes_closed': True, 'execution_lock_sha256': sha(cohort / 'EXECUTION_LOCK.json'),
        'source_map_sha256': lr.PINNED_SOURCE, 'case_manifest_sha256': sha(ledger_path),
        'absolute_deadline_unix': 2, 'completed_unix': 1, 'analysis_sha256': sha(evaluation / 'ANALYSIS.json'),
        'retained_record_audit_sha256': sha(evaluation / 'RETAINED_RECORD_AUDIT.json'),
        'evaluated_episodes': len(cases), 'evaluation_joint_steps': len(cases) * 256, 'evaluation_agent_decisions': len(cases) * 1024}
    put(cohort / 'STUDY_COMPLETE.json', final)
    return cohort, selection_path, ledger_path

