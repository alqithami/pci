#!/usr/bin/env python3
"""Verify public warehouse sufficient statistics without running experiments."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/warehouse_development_20261106'
SOURCE = ROOT / 'experiments/PCI_M3_Fair_Training_v1'
COUNTS = ('physical_replacements', 'institutional_replacements', 'total_replacements',
          'deliveries', 'pickups', 'joint_steps')
SOURCE_MAP = 'e7fbf20f649ce40a21d42d644756cf519dd6fd2717564c5bbcf0eb0793473ed7'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path):
    return json.loads(path.read_bytes())


def keyed_csv(name, fields):
    with (DATA / name).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    index = {tuple(r[k] for k in fields): r for r in rows}
    require(len(index) == len(rows), 'Duplicate coordinate in ' + name)
    return index


def main():
    manifest = json_file(DATA / 'PUBLIC_RELEASE_MANIFEST.json')
    for rel, expected in manifest['files'].items():
        p = ROOT / rel
        require(p.resolve().is_relative_to(ROOT), 'Unsafe release path')
        require(p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'],
                'Release member mismatch: ' + rel)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE / 'protocol'))
    import build_study as study
    import frontier
    sources = study.source_map()
    require(len(sources) == 43 and study.digest(sources) == SOURCE_MAP, 'Source closure mismatch')
    require(json_file(SOURCE / 'protocol/PROTOCOL_DRAFT.json') == study.protocol(), 'Protocol body mismatch')
    finish = json_file(DATA / 'STUDY_COMPLETE.json')
    evidence = json_file(DATA / 'EVIDENCE_SUMMARY.json')
    require(finish['stage'] == 'development' and finish['status'] == 'COMPLETE_AUDITED_COHORT'
            and finish['source_map_sha256'] == SOURCE_MAP
            and finish['trained_policies'] == 144 and finish['evaluated_episodes'] == 5760
            and finish['evaluation_joint_steps'] == 1474560
            and finish['evaluation_agent_decisions'] == 5898240, 'Completion scope mismatch')
    require(evidence['retained_record_audit']['status'] == 'PASS'
            and evidence['retained_record_audit']['original_sha256'] == finish['retained_record_audit_sha256'],
            'Audit summary binding mismatch')

    candidates = {x['candidate_id']: x for x in study.candidates()}
    seeds = [str(s) for s in study.DEVELOPMENT_SEEDS]
    profiles = tuple(sorted(study.profiles()))
    pairs = {(c, s) for c in candidates for s in seeds}
    triples = {(c, s, p) for c, s in pairs for p in profiles}
    seed_rows = keyed_csv('candidate_seed_points.csv', ('candidate_id', 'training_seed'))
    profile_rows = keyed_csv('profile_seed_counts.csv', ('candidate_id', 'training_seed', 'profile_id'))
    work = keyed_csv('deployment_oracle_work.csv', ('candidate_id', 'training_seed', 'profile_id'))
    require(set(seed_rows) == pairs and set(profile_rows) == set(work) == triples, 'Incomplete development grid')

    declared = json_file(DATA / 'development_selection_rows.json')
    raw = {(r['candidate_id'], str(r['training_seed'])): r for r in declared}
    require(len(declared) == len(raw) == 144 and set(raw) == pairs, 'Incomplete selection inputs')
    for c, s in pairs:
        family = candidates[c]['family']
        totals = dict.fromkeys(COUNTS, 0)
        for p in profiles:
            r = profile_rows[c, s, p]
            q = work[c, s, p]
            require(r['family'] == q['family'] == family and r['domain'] == q['domain'] == 'warehouse'
                    and r['n_agents'] == q['n_agents'] == '4', 'Profile identity mismatch')
            require(int(r['joint_steps']) == int(q['joint_steps']) == 2560, 'Profile exposure mismatch')
            for k in COUNTS:
                value = int(r[k])
                require(value >= 0, 'Negative count')
                totals[k] += value
            a, b, t = (int(r[k]) for k in COUNTS[:3])
            require(abs(a - b) <= t <= a + b and max(a, b, t) <= 10240, 'Invalid correction counts')
            require(family != 'shieldaware' or a == b == t == 0, 'Nonzero shield corrections')
            expected = (625 * 2560, 2560) if family == 'shieldaware' else (0, 0)
            require((int(q['requested_rows']), int(q['callback_calls'])) == expected, 'Oracle work mismatch')
        r = seed_rows[c, s]
        d = raw[c, s]
        require(r['family'] == d['family'] == family and all(int(r[k]) == d[k] == totals[k] for k in COUNTS),
                'Seed total mismatch')
        exposure = 4 * totals['joint_steps']
        require(d['normalized_cost'] == totals['institutional_replacements'] / exposure
                and d['normalized_utility'] == totals['deliveries'] / exposure,
                'Endpoint normalization mismatch')

    selection = frontier.select(declared, source_sha256=sha(DATA / 'development_selection_rows.json'))
    require(selection == json_file(DATA / 'DEVELOPMENT_SELECTION.json'), 'Frozen selection mismatch')
    means = keyed_csv('candidate_means.csv', ('candidate_id',))
    require(set(means) == {(c,) for c in candidates}, 'Candidate means membership mismatch')
    hv = keyed_csv('frozen_selected_pair_hypervolumes.csv', ('family', 'training_seed'))
    require(set(hv) == {(f, s) for f in study.FAMILIES for s in seeds}, 'Hypervolume rows mismatch')
    for f, d in selection['families'].items():
        for c, point in d['all_candidate_mean_points'].items():
            r = means[c,]
            require(r['family'] == f and math.isclose(float(r['institutional_per_1000_agent_decisions']), 1000 * point['cost'])
                    and math.isclose(float(r['deliveries_per_1000_joint_steps']), 4000 * point['utility'])
                    and (r['selected_for_confirmation'] == 'True') == (c in d['selected']), 'Candidate mean mismatch')
        values = [frontier.hypervolume([(raw[c, s]['normalized_cost'], raw[c, s]['normalized_utility'])
                                         for c in d['selected']], cost_limit=selection['cost_limit']) for s in seeds]
        for s, value in zip(seeds, values):
            r = hv[f, s]
            require([r['candidate_1'], r['candidate_2']] == d['selected']
                    and float(r['selected_pair_seed_hypervolume']) == value
                    and float(r['selection_hypervolume_of_mean_points']) == d['development_hypervolume']
                    and float(r['mean_of_selected_pair_seed_hypervolumes']) == sum(values) / len(values),
                    'Selected-pair hypervolume mismatch')
    print(json.dumps({'status': 'PASS_PUBLIC_SOURCE_AND_DEVELOPMENT_SUMMARIES',
                      'policies': 144, 'profiles': 576, 'episodes_in_original_audit': 5760,
                      'selected_candidate_ids': selection['selected_candidate_ids'],
                      'raw_trace_replay_performed': False}))


if __name__ == '__main__':
    main()
