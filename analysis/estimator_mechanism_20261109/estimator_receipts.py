"""Read-only estimator-cohort receipt adapter; never replays raw models or traces.

The caller supplies a complete closed estimator cohort and its original input ledger.
All hashes authenticate internal record bindings, not an external publisher.
Selection provenance is a mandatory raw-hash binding in the prospectively issued
SOURCE_DESIGN_LOCK; this adapter does not infer that selection from test data.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types

PINNED_SOURCE = 'e7fbf20f649ce40a21d42d644756cf519dd6fd2717564c5bbcf0eb0793473ed7'
COUNTS = ('physical_replacements', 'institutional_replacements',
          'total_replacements', 'deliveries', 'pickups', 'joint_steps')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


class Snapshot:
    """Hash and parse one held byte body, then recheck every accessed input."""
    def __init__(self):
        self.files = {}

    @staticmethod
    def stamp(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)

    def fresh(self, path):
        path = Path(path).absolute()
        need(path.resolve() == path, 'SYMLINK_INPUT:' + str(path))
        before = path.lstat()
        need(stat.S_ISREG(before.st_mode), 'NONREGULAR_INPUT:' + str(path))
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            first = os.fstat(fd)
            need(self.stamp(first) == self.stamp(before), 'INPUT_CHANGED_BEFORE_READ')
            with os.fdopen(fd, 'rb', closefd=False) as handle:
                body = handle.read()
            last = os.fstat(fd)
            after = path.lstat()
            need(self.stamp(first) == self.stamp(last) == self.stamp(after)
                 and len(body) == first.st_size, 'INPUT_CHANGED_DURING_READ')
        finally:
            os.close(fd)
        return body, self.stamp(last)

    def body(self, path):
        path = Path(path).absolute()
        if path not in self.files:
            body, stamp = self.fresh(path)
            self.files[path] = (body, stamp, hashlib.sha256(body).hexdigest())
        return self.files[path][0]

    def sha(self, path):
        return hashlib.sha256(self.body(path)).hexdigest()

    def json(self, path, schema=None):
        result = json.loads(self.body(path), object_pairs_hook=_unique,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_JSON:' + x)))
        if schema:
            need(isinstance(result, dict) and result.get('schema') == schema,
                 'PRODUCER_SCHEMA:' + Path(path).name)
        return result

    def check(self):
        for path, (body, stamp, _) in self.files.items():
            current, after = self.fresh(path)
            need(current == body and stamp == after, 'INPUT_CHANGED_AFTER_READ:' + str(path))


def _protocol(source_root, snapshot):
    source_root = Path(source_root).absolute()
    sources = {p.relative_to(source_root).as_posix(): snapshot.sha(p)
               for folder in ('runtime', 'protocol')
               for p in sorted((source_root / folder).rglob('*.py'))
               if '__pycache__' not in p.parts}
    need(len(sources) == 43 and digest(sources) == PINNED_SOURCE, 'UNQUALIFIED_SOURCE_MAP')
    # Compile the exact held bytes already hashed; do not run an unbound loader.
    old_bridge = sys.modules.get('bridge_design')
    modules = {}
    try:
        for name in ('bridge_design', 'build_study'):
            path = source_root / 'protocol' / (name + '.py')
            module = types.ModuleType(name)
            module.__file__ = str(path)
            if name == 'bridge_design':
                sys.modules[name] = module
            exec(compile(snapshot.body(path), str(path), 'exec'), module.__dict__)
            modules[name] = module
    finally:
        if old_bridge is None:
            sys.modules.pop('bridge_design', None)
        else:
            sys.modules['bridge_design'] = old_bridge
    return sources, modules['build_study']


def _index(rows, expected, fields, label):
    need(type(rows) is list, 'LIST_REQUIRED:' + label)
    keys = [tuple(row[k] for k in fields) for row in rows]
    need(len(keys) == len(set(keys)) and set(keys) == set(expected),
         'INCOMPLETE_OR_DUPLICATE:' + label)
    return dict(zip(keys, rows))


def _safe_relative(value):
    path = Path(value)
    need(not path.is_absolute() and path.parts and '..' not in path.parts, 'UNSAFE_RELATIVE_PATH')
    return path


def load_learning_receipts(cohort, source_root, selection_path, case_manifest_path,
                           *, source_design_lock_path=None, _test_only_synthetic=False):
    """Return complete estimator counts plus validated receipt provenance.

    SOURCE_DESIGN_LOCK must contain development_selection_sha256, the raw hash
    prospectively frozen before these confirmation outcomes. Its stage scope is
    reviewed externally; this adapter enforces source/protocol/hash bindings.
    This separate adapter accepts only the declared warehouse estimator stage.
    No live results are opened unless a
    caller explicitly invokes this function with their paths.
    """
    synthetic = (Path(cohort) / 'SYNTHETIC_TEST_ONLY.json').exists()
    need(not synthetic or _test_only_synthetic, 'SYNTHETIC_COHORT_REFUSED_FOR_RESEARCH_ANALYSIS')
    snapshot = Snapshot()
    result = _load(cohort, source_root, selection_path, case_manifest_path,
                   source_design_lock_path, snapshot)
    snapshot.check()
    result['provenance']['input_sha256'] = {str(p): value[2] for p, value in snapshot.files.items()}
    result['provenance']['input_stability'] = 'PASS_HELD_BODY_HASH_PARSE_AND_FINAL_RECHECK'
    result['provenance']['synthetic_validation_only'] = synthetic
    return result


def _load(cohort, source_root, selection_path, case_manifest_path, design_path, s):
    c = Path(cohort).absolute()
    e = c / 'evaluation'
    ledger_path = Path(case_manifest_path).absolute()
    input_root = ledger_path.parent.parent
    design_path = Path(design_path).absolute() if design_path else input_root / 'SOURCE_DESIGN_LOCK.json'
    sources, study = _protocol(source_root, s)
    protocol = study.protocol()
    protocol_hash = digest(protocol)
    final = s.json(c / 'STUDY_COMPLETE.json', 'PCI-FAIR-COHORT-COMPLETE-v1')
    need(final.get('status') == 'COMPLETE_AUDITED_COHORT'
         and final.get('stage') == 'estimator_ablation', 'COMPLETE_ESTIMATOR_COHORT_REQUIRED')
    domain = final.get('study_domain')
    need(domain == 'warehouse', 'WAREHOUSE_ESTIMATOR_DOMAIN_REQUIRED')
    seeds = study.ESTIMATOR_LEARNING_SEEDS
    development_seeds = study.DEVELOPMENT_SEEDS
    need(final.get('all_owned_worker_processes_closed') is True, 'OWNED_WORKERS_NOT_CLOSED')
    need(final.get('trained_policies') == 60 and final.get('training_joint_steps') == 31457280,
         'PRESCRIBED_TRAINING_BUDGET')
    ex = s.json(c / 'EXECUTION_LOCK.json')
    need(final['execution_lock_sha256'] == s.sha(c / 'EXECUTION_LOCK.json'), 'EXECUTION_LOCK_HASH')
    need(ex['source_sha256'] == sources and final['source_map_sha256'] == PINNED_SOURCE, 'SOURCE_BINDING')
    cfg = s.json(input_root / 'CONTROLLER_CONFIG.json', 'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1')
    need(ex['controller_configuration'] == cfg
         and ex['controller_configuration_file_sha256'] == s.sha(input_root / 'CONTROLLER_CONFIG.json'),
         'CONTROLLER_INPUT_BINDING')
    need(cfg['role'] == 'scientific' and cfg['stage'] == 'estimator_ablation'
         and cfg['study_domain'] == domain and cfg['device'] == 'cuda'
         and cfg['worlds_per_profile'] == 20, 'CONTROLLER_CONFIRMATION_SCOPE')
    need(final['absolute_deadline_unix'] == cfg['deadline_unix']
         and final['completed_unix'] <= cfg['deadline_unix'], 'DEADLINE_BINDING')
    ledger = s.json(ledger_path, 'PCI-COMPLETE-CASE-MANIFEST-v1')
    need(ledger['stage'] == 'estimator_ablation' and ledger['study_domain'] == domain
         and ledger['case_count'] == 60 and ledger['protocol_canonical_sha256'] == protocol_hash,
         'TRAINING_LEDGER_SCOPE')
    need(final['case_manifest_sha256'] == ex['case_manifest_sha256'] == s.sha(ledger_path), 'TRAINING_LEDGER_HASH')
    need(s.json(ledger_path.parent / 'PROTOCOL_DRAFT.json') == protocol, 'PROTOCOL_BODY')
    design = s.json(design_path, 'PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1')
    need(design.get('scientific_launch_authorized') is True and design.get('engineering_complete') is True,
         'SOURCE_DESIGN_AUTHORITY')
    need(design['source_sha256'] == sources and design['source_map_sha256'] == PINNED_SOURCE
         and design['protocol_canonical_sha256'] == protocol_hash, 'SOURCE_DESIGN_BINDING')
    need(design.get('development_selection_sha256') == s.sha(selection_path),
         'PROSPECTIVELY_BOUND_DEVELOPMENT_SELECTION_REQUIRED')
    need(design.get('stage') == 'estimator_ablation' and design.get('study_domain') == 'warehouse'
         and design.get('training_policies') == 60 and design.get('training_seeds') == list(seeds),
         'EXPLICIT_ESTIMATOR_DESIGN_SCOPE_REQUIRED')
    analysis_root = Path(__file__).resolve().parent
    need(design.get('estimator_analysis_recipe_sha256') == s.sha(analysis_root / 'ESTIMATOR_ANALYSIS_RECIPE.json'),
         'PROSPECTIVE_ESTIMATOR_ANALYSIS_RECIPE_BINDING_REQUIRED')
    need(design.get('estimator_analysis_implementation_lock_sha256') ==
         s.sha(analysis_root / 'ESTIMATOR_ANALYSIS_IMPLEMENTATION_LOCK.json'),
         'PROSPECTIVE_ESTIMATOR_ANALYSIS_IMPLEMENTATION_BINDING_REQUIRED')
    selection = s.json(selection_path, 'PCI-DEVELOPMENT-FRONTIER-SELECTION-v1')
    need(selection['status'] == 'DEVELOPMENT_ONLY_SELECTION'
         and selection['study_domain'] == domain and selection['development_seeds'] == development_seeds
         and selection['primary_test_data_opened'] is False
         and selection['cost_limit'] == study.COST_LIMITS[domain], 'DEVELOPMENT_SELECTION_SCOPE')
    candidates = {row['candidate_id']: row for row in study.candidates()}
    selected = selection['selected_candidate_ids']
    need(type(selected) is list and len(selected) == len(set(selected)) == 8
         and all(key in candidates for key in selected), 'EIGHT_UNIQUE_SELECTED_CANDIDATES')
    for family in study.FAMILIES:
        ids = [key for key in selected if candidates[key]['family'] == family]
        need(len(ids) == 2 and selection['families'][family]['selected'] == ids, 'TWO_SELECTED_PER_FAMILY')
    base_ids = selection.get('estimator_ablation_candidate_ids')
    need(base_ids == ['direct_02', 'direct_11'], 'FROZEN_NONZERO_ABLATION_PAIR_REQUIRED')
    need(all(candidates[key]['family'] == 'direct' and candidates[key]['auxiliary_coef'] > 0
             for key in base_ids), 'NONZERO_DIRECT_ABLATION_ONLY')
    candidates = {key + '_' + mode: dict(candidates[key], candidate_id=key + '_' + mode,
                   risk_estimator=mode) for key in base_ids for mode in ('exact', 'score', 'conditional')}
    selected = list(candidates)
    coordinates = {(key, seed) for key in selected for seed in seeds}
    scheduled = _index(ledger['cases'], coordinates, ('candidate_id', 'training_seed'), 'TRAINING_CASES')
    expected_cells = {(domain, 4, key, seed) for key, seed in coordinates}
    declared_cells = []
    for cell in cfg['expected_cells']:
        declared_cells += [(cell['domain'], cell['n_agents'], key, seed)
                           for key in cell['candidate_ids'] for seed in cell['training_seeds']]
    need(len(declared_cells) == len(set(declared_cells)) and set(declared_cells) == expected_cells,
         'CONTROLLER_EXPECTED_CELLS')
    launch = s.json(input_root / 'LAUNCH_LOCK.json', 'PCI-FAIR-SCIENTIFIC-LAUNCH-v1')
    ready = s.json(input_root / 'CUDA_READINESS.json', 'PCI-FAIR-CUDA-READINESS-v1')
    need(launch.get('scientific_launch_authorized') is True and ready.get('status') == 'PASS'
         and ready.get('all_required_methods_pass') is True, 'LAUNCH_READINESS')
    need(ready.get('estimator_modes_pass') == {'exact': True, 'score': True, 'conditional': True},
         'ALL_ESTIMATOR_MODES_READINESS_REQUIRED')
    expected = {'controller_config_sha256': digest(cfg), 'case_manifest_sha256': s.sha(ledger_path),
                'source_design_lock_sha256': s.sha(design_path),
                'readiness_receipt_sha256': s.sha(input_root / 'CUDA_READINESS.json'),
                'source_map_sha256': PINNED_SOURCE, 'out': cfg['out'], 'deadline_unix': cfg['deadline_unix']}
    need(all(launch.get(k) == v == ex['authority'].get(k) for k, v in expected.items()), 'LAUNCH_EXECUTION_BINDINGS')
    need(ready['source_map_sha256'] == PINNED_SOURCE and ready['device'] == 'cuda'
         and launch['host_boot_id'] == ready['host_boot_id'] == ex['authority']['host_boot_id']
         and launch['gpu_uuid'] == ready['gpu_uuid'] == ex['authority']['gpu_uuid'], 'RECORDED_HOST_DEVICE_BINDING')
    external = {cfg['source_design_lock']: s.sha(design_path), cfg['launch_lock']: s.sha(input_root / 'LAUNCH_LOCK.json'),
                cfg['readiness_receipt']: s.sha(input_root / 'CUDA_READINESS.json'),
                cfg['protocol_file']: s.sha(ledger_path.parent / 'PROTOCOL_DRAFT.json')}
    need(ex['authority']['external_record_sha256'] == external, 'AUTHORITY_EXTERNAL_RECORDS')
    native = None
    configs = {}
    for (candidate, seed), row in scheduled.items():
        path = ledger_path.parent / _safe_relative(row['path'])
        value = s.json(path, 'PCI-FAIR-TRAIN-CONFIG-v1')
        need(s.sha(path) == row['sha256'], 'TRAINING_CONFIG_HASH')
        expected_cfg = study.train_config(candidates[candidate], seed, seeds.index(seed), role='confirmatory',
            n=4, domain=domain, steps=524288, lock_hash=protocol_hash, stage='estimator_ablation',
            domain_profiles=native['profiles'] if native else None,
            domain_slots=native.get('env_profile_ids') if native else None)
        need(value == expected_cfg and row['case_id'] == value['run_id'], 'TRAINING_CONFIG_RECONSTRUCTION')
        need(row['family'] == value['method'] and row['domain'] == domain and row['n_agents'] == 4
             and row['run_role'] == 'confirmatory' and row['total_steps'] == 524288
             and row['risk_estimator'] == value['risk_estimator']
             and row['actual_auxiliary_coef'] == value['auxiliary_coef'], 'TRAINING_CASE_DESCRIPTOR')
        configs[row['case_id']] = value
    manifest = s.json(e / 'MANIFEST.json', 'PCI-FAIR-EVALUATION-MANIFEST-v1')
    need(manifest == s.json(c / 'EVALUATION_MANIFEST.json', 'PCI-FAIR-EVALUATION-MANIFEST-v1'), 'EVALUATION_MANIFEST_COPIES')
    need(manifest['training_case_manifest'] == ledger and manifest['training_case_manifest_sha256'] == digest(ledger), 'EVALUATION_TRAINING_LEDGER')
    need(manifest['stage'] == 'estimator_ablation' and manifest['run_role'] == 'confirmatory'
         and manifest['protocol_sha256'] == protocol_hash and manifest['worlds_per_profile'] == 20
         and manifest['training_seed_clusters'] == seeds, 'EVALUATION_SCOPE')
    evaluation_sources = {key.removeprefix('runtime/'): value for key, value in sources.items()
                          if key.startswith(('runtime/training/', 'runtime/evaluation_fair/'))}
    need(manifest['source_sha256'] == evaluation_sources, 'EVALUATION_SOURCE_MAP')
    policies = _index(manifest['policies'], coordinates, ('candidate_id', 'training_seed'), 'EVALUATION_POLICIES')
    expected_cases = {}
    profile_definitions = None
    for (candidate, seed), policy in policies.items():
        run = scheduled[candidate, seed]['case_id']
        value = configs[run]
        need(policy['directory'] == policy['run_id'] == run and policy['family'] == value['method']
             and policy['domain'] == domain and policy['n_agents'] == 4, 'POLICY_DESCRIPTOR')
        if profile_definitions is None:
            profile_definitions = value['profiles']
        need(value['profiles'] == profile_definitions, 'COMMON_TRAINED_PROFILE_DESIGN')
        for profile, environment_config in sorted(value['profiles'].items()):
            for world in range(20):
                key = f'{run}__{profile}__w{world:02d}'
                world_seed = 3804000000 + 1000 * seeds.index(seed) + world
                expected_cases[key] = {'case_id': key, 'policy_directory': run, 'candidate_id': candidate,
                    'family': value['method'], 'method': value['method'], 'training_seed': seed,
                    'domain': domain, 'n_agents': 4, 'profile_id': profile,
                    'environment_config': environment_config, 'world_index': world,
                    'world_seed': world_seed, 'policy_seed': world_seed + 50000000}
    cases = _index(manifest['cases'], [(key,) for key in expected_cases], ('case_id',), 'EVALUATION_CASES')
    need(all(row == expected_cases[key[0]] for key, row in cases.items()), 'EVALUATION_CASE_RECONSTRUCTION')
    episode_count = len(expected_cases)
    joint_steps = sum(case['environment_config']['horizon'] for case in expected_cases.values())
    need(manifest['complete_policy_count'] == 60 and manifest['complete_episode_count'] == episode_count,
         'EVALUATION_COHORT_COUNTS')
    manifest_hash = digest(manifest)
    complete = s.json(e / 'EVALUATION_COMPLETE.json', 'PCI-FAIR-EVALUATION-COMPLETE-v1')
    audit = s.json(e / 'RETAINED_RECORD_AUDIT.json', 'PCI-FAIR-RETAINED-RECORD-AUDIT-v1')
    analysis = s.json(e / 'ANALYSIS.json', 'PCI-FAIR-EVALUATION-AGGREGATE-v1')
    need(complete['status'] == 'COMPLETE' and audit['status'] == 'PASS', 'COMPLETE_EVALUATION_AND_AUDIT')
    need(complete['manifest_sha256'] == audit['manifest_sha256'] == manifest_hash, 'MANIFEST_BINDING')
    need(final['analysis_sha256'] == complete['analysis_sha256'] == s.sha(e / 'ANALYSIS.json'), 'ANALYSIS_HASH')
    need(final['retained_record_audit_sha256'] == s.sha(e / 'RETAINED_RECORD_AUDIT.json'), 'AUDIT_HASH')
    need(complete['all_policy_count'] == audit['policy_count'] == 60
         and complete['all_episode_count'] == audit['episode_count'] == final['evaluated_episodes'] == episode_count,
         'CLOSED_COHORT_COUNTS')
    need(complete['joint_steps'] == audit['joint_steps'] == final['evaluation_joint_steps'] == joint_steps
         and complete['agent_decisions'] == audit['agent_decisions'] == final['evaluation_agent_decisions'] == 4 * joint_steps,
         'CLOSED_COHORT_EXPOSURE')
    for flag in ('independent_numpy_actor', 'independent_scalar_stage_checks', 'recorded_action_native_replay'):
        need(audit[flag] is True, 'AUDIT_METHOD_FLAG')
    need(audit['replacement_policy_rollouts'] == 0, 'REPLACEMENT_POLICY_ROLLOUTS')
    reports = _index(audit['case_reports'], [(key,) for key in expected_cases], ('case_id',), 'AUDIT_CASES')
    expected_names = {key + '.json' for key in expected_cases}
    need({path.name for path in (e / 'cases').iterdir()} == expected_names, 'CASE_FILE_MEMBERSHIP')
    need(set(complete['case_receipts']) == set(complete['trace_intent_receipts']) == set(expected_cases), 'CASE_RECEIPT_MEMBERSHIP')
    by_profile = defaultdict(list)
    for key, case in expected_cases.items():
        path = e / 'cases' / (key + '.json')
        body = s.json(path, 'PCI-FAIR-EVALUATION-CASE-v1')
        report = reports[key,]
        need(s.sha(path) == complete['case_receipts'][key], 'CASE_RECEIPT_HASH')
        need(body['status'] == 'COMPLETE' and body['case'] == case and body['manifest_sha256'] == manifest_hash,
             'CASE_DESCRIPTOR')
        horizon = case['environment_config']['horizon']
        need(report['status'] == 'PASS' and report['trace_sha256'] == body['trace_seal']['sha256']
             and report['joint_steps'] == body['totals']['joint_steps'] == horizon
             and report['agent_decisions'] == 4 * horizon, 'CASE_AUDIT_EXPOSURE')
        need(body['trace_seal']['publication_intent_sha256'] == complete['trace_intent_receipts'][key]
             and body['trace_seal']['records'] == horizon + 2, 'CASE_TRACE_BINDING')
        policy = policies[case['candidate_id'], case['training_seed']]
        need(body['training_signature'] == policy['training_signature']
             and body['portable_export_sha256'] == policy['portable_export_sha256'], 'CASE_POLICY_BINDING')
        for name in COUNTS:
            count = body['totals'][name]
            need(type(count) is int and count >= 0, 'INVALID_CASE_COUNT:' + name)
        physical, institutional, total = (body['totals'][name] for name in COUNTS[:3])
        need(max(physical, institutional, total, body['totals']['deliveries']) <= 4 * horizon
             and abs(physical - institutional) <= total <= physical + institutional, 'COUNT_EXPOSURE_OR_HAMMING_BOUND')
        if case['family'] == 'shieldaware':
            need(physical == institutional == total == 0, 'SHIELD_REPLACEMENT_INVARIANT')
        expected_queries = (625 * horizon, horizon) if case['family'] == 'shieldaware' else (0, 0)
        need((body['deployment_policy_requested_rows'], body['deployment_policy_callback_calls']) == expected_queries,
             'PINNED_N4_POLICY_QUERY_EXPOSURE')
        by_profile[case['candidate_id'], case['training_seed'], case['profile_id']].append(body)
    profile_rows, seed_rows = [], []
    profiles = sorted(profile_definitions)
    for (candidate, seed), policy in policies.items():
        identity = {key: policy[key] for key in ('candidate_id', 'family', 'training_seed', 'domain', 'n_agents')}
        per_profile = []
        for profile in profiles:
            summaries = by_profile[candidate, seed, profile]
            need(len(summaries) == 20, 'PROFILE_WORLD_COUNT')
            counts = {name: sum(row['totals'][name] for row in summaries) for name in COUNTS}
            profile_rows.append({**identity, 'profile_id': profile, **counts})
            per_profile.append(counts)
        need(len({row['joint_steps'] for row in per_profile}) == 1, 'UNBALANCED_PROFILE_EXPOSURE')
        counts = {name: sum(row[name] for row in per_profile) for name in COUNTS}
        steps = counts['joint_steps']
        seed_rows.append({**identity, **counts, 'cohort_complete': True, 'evaluation_complete': True,
            'profile_weights': {profile: 1 / len(profiles) for profile in profiles},
            'institutional_per_1000_agent_decisions': 1000 * counts['institutional_replacements'] / (4 * steps),
            'deliveries_per_1000_joint_steps': 1000 * counts['deliveries'] / steps,
            'normalized_cost': counts['institutional_replacements'] / (4 * steps),
            'normalized_utility': counts['deliveries'] / (4 * steps)})
    need(analysis['status'] == 'PASS_COMPLETE_COHORT' and analysis['stage'] == 'estimator_ablation'
         and analysis['policy_count'] == 60 and analysis['episode_count'] == episode_count, 'ANALYSIS_SCOPE')
    need(_index(analysis['equal_profile_seed_rows'], coordinates, ('candidate_id', 'training_seed'), 'ANALYSIS_SEED_ROWS')
         == _index(seed_rows, coordinates, ('candidate_id', 'training_seed'), 'RECOMPUTED_SEED_ROWS'), 'SEED_AGGREGATE_RECOMPUTATION')
    profile_keys = {(key, seed, profile) for key, seed in coordinates for profile in profiles}
    need(_index(analysis['profile_seed_rows'], profile_keys, ('candidate_id', 'training_seed', 'profile_id'), 'ANALYSIS_PROFILE_ROWS')
         == _index(profile_rows, profile_keys, ('candidate_id', 'training_seed', 'profile_id'), 'RECOMPUTED_PROFILE_ROWS'),
         'PROFILE_AGGREGATE_RECOMPUTATION')
    need({path.name for path in (e / 'cases').iterdir()} == expected_names, 'CASE_MEMBERSHIP_CHANGED')
    return {'schema': 'PCI-ESTIMATOR-LEARNING-RECEIPT-ADAPTER-v1',
        'status': 'PASS_COMPLETE_ESTIMATOR_RECEIPT_BINDINGS', 'domain': domain,
        'selection': selection, 'seed_rows': seed_rows, 'profile_rows': profile_rows,
        'provenance': {'source_map_sha256': PINNED_SOURCE, 'protocol_canonical_sha256': protocol_hash,
            'development_selection_sha256': s.sha(selection_path), 'source_design_lock_sha256': s.sha(design_path),
            'analysis_recipe_sha256': s.sha(analysis_root / 'ESTIMATOR_ANALYSIS_RECIPE.json'),
            'analysis_implementation_lock_sha256': s.sha(analysis_root / 'ESTIMATOR_ANALYSIS_IMPLEMENTATION_LOCK.json'),
            'case_manifest_sha256': s.sha(ledger_path), 'study_complete_sha256': s.sha(c / 'STUDY_COMPLETE.json'),
            'retained_record_audit_sha256': s.sha(e / 'RETAINED_RECORD_AUDIT.json'),
            'scope': 'Source/configuration and published replay-audit receipt bindings plus independently recomputed complete case-summary aggregation; raw models/traces are not independently replayed by this adapter. Prospective authorization scope review remains external.'}}
