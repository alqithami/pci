#!/usr/bin/env python3
"""Check the public warehouse confirmation projection without third-party packages.

This reconstructs exact endpoints and all paired sign tests from the published
policy counts. It does not replay policies or validate private retained traces,
bootstrap RNG streams, simulator fidelity, data origin, or the full Holm12 family.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REL = Path('results/warehouse_confirmation_20261109')
SEEDS = tuple(range(62001, 62021))
FAMILIES = ('fixed', 'direct', 'lagrangian', 'shieldaware')
IDS = {
    'fixed': ('fixed_05', 'fixed_10'),
    'direct': ('direct_02', 'direct_11'),
    'lagrangian': ('lagrangian_03', 'lagrangian_11'),
    'shieldaware': ('shieldaware_00', 'shieldaware_04'),
}
NAMES = {
    'fixed': 'Warehouse Direct minus Fixed',
    'lagrangian': 'Warehouse Direct minus Lagrangian',
    'shieldaware': 'Warehouse Direct minus shield-aware',
}


def need(value, reason):
    if not value:
        raise AssertionError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ratio(item):
    return Fraction(item['numerator'], item['denominator'])


def area(points, tau):
    # Independent two-rectangle inclusion/exclusion, rather than frontier sweep.
    need(len(points) == 2, 'Exactly two frozen candidates required')
    a, b = [(c, u) if c <= tau else (tau, Fraction(0)) for c, u in points]
    return ((tau - a[0]) * a[1] + (tau - b[0]) * b[1]
            - (tau - max(a[0], b[0])) * min(a[1], b[1])) / tau


def exact_p(differences):
    common = math.lcm(*(x.denominator for x in differences))
    values = [x.numerator * (common // x.denominator) for x in differences]
    target = abs(sum(values))
    sums = [0]
    for value in values:
        sums = [s + value for s in sums] + [s - value for s in sums]
    need(len(sums) == 2**20, 'Sign-assignment count')
    exceedances = sum(abs(s) >= target for s in sums)
    return Fraction(exceedances, len(sums)), exceedances


def main():
    directory = ROOT / REL
    manifest = read(directory / 'MANIFEST.json')
    need(manifest['schema'] == 'PCI-PUBLIC-CONFIRMATION-MANIFEST-v1', 'Manifest schema')
    for name, entry in manifest['files'].items():
        path = ROOT / name
        need(path.is_file() and path.stat().st_size == entry['bytes'], 'Missing/size: ' + name)
        need(sha(path) == entry['sha256'], 'SHA256: ' + name)

    packet = read(directory / 'PUBLIC_RESULT.json')
    need(packet['schema'] == 'PCI-PUBLIC-CONFIRMATION-PROJECTION-v1', 'Projection schema')
    need(packet['status'] == 'DESCRIPTIVE_PARTIAL_FAMILY', 'Partial family status')
    need(packet['family_complete'] is False and len(packet['pending_contrasts']) == 9,
         'Nine prospective contrasts remain pending')
    need(all('adjusted_p_value' not in c and 'reject' not in c for c in packet['contrasts']),
         'No completed multiplicity decisions allowed')

    bindings = read(directory / 'SOURCE_BINDINGS.json')
    source = ROOT / 'experiments/PCI_M3_Fair_Training_v1'
    need(len(bindings['source_sha256']) == 43, 'Frozen source count')
    for name, expected in bindings['source_sha256'].items():
        need(sha(source / name) == expected, 'Frozen study source: ' + name)
    canonical = json.dumps(bindings['source_sha256'], sort_keys=True,
                           separators=(',', ':')).encode()
    need(hashlib.sha256(canonical).hexdigest() == packet['source_map_sha256'],
         'Canonical source map')
    for name, expected in bindings['analysis_files_sha256'].items():
        need(sha(ROOT / 'analysis/confirmation_20261106' / name) == expected,
             'Frozen analysis file: ' + name)

    endpoints = packet['endpoint_data']
    need(ratio(endpoints['cost_limit']) == Fraction(1, 20), 'Primary cost rectangle')
    groups = defaultdict(list)
    seen = set()
    for row in endpoints['policy_points']:
        key = row['training_seed'], row['candidate_id']
        need(key not in seen, 'Duplicate policy point')
        seen.add(key)
        need(row['training_seed'] in SEEDS and row['family'] in FAMILIES, 'Point namespace')
        need(row['candidate_id'] in IDS[row['family']], 'Frozen candidate identity')
        need(row['n_agents'] == 4 and row['joint_steps'] == 20480, 'Policy episode denominator')
        c = Fraction(row['institutional_replacements'], 4 * row['joint_steps'])
        u = Fraction(row['deliveries'], 4 * row['joint_steps'])
        need(c == ratio(row['cost']) and u == ratio(row['utility']), 'Count normalization')
        groups[row['training_seed'], row['family']].append((c, u))
    need(len(seen) == 160 and len(groups) == 80, 'Complete policy/endpoint count')
    tau = Fraction(1, 20)
    calculated = {(seed, family): area(groups[seed, family], tau)
                  for seed in SEEDS for family in FAMILIES}
    retained = {(x['training_seed'], x['family']): ratio(x['hypervolume'])
                for x in endpoints['family_cluster_hypervolumes']}
    need(calculated == retained, 'All eighty seed-family areas')
    contrasts = {x['contrast_id']: x for x in packet['contrasts']}
    need(set(contrasts) == set(NAMES.values()), 'All three warehouse contrasts')
    results = []
    for reference, name in NAMES.items():
        record = contrasts[name]
        values = [calculated[s, 'direct'] - calculated[s, reference] for s in SEEDS]
        need([x['training_seed'] for x in record['cluster_differences']] == list(SEEDS),
             'Contrast seed order')
        need(values == [ratio(x['difference']) for x in record['cluster_differences']],
             'Twenty exact differences: ' + name)
        mean = sum(values) / 20
        variance = sum((x - mean)**2 for x in values) / 19
        p, count = exact_p(values)
        need(mean == ratio(record['mean_effect']), 'Paired mean: ' + name)
        need(variance == ratio(record['paired_sample_variance']), 'Paired variance: ' + name)
        need(p == ratio(record['p_value']) == ratio(record['test']['p_value']), 'Exact p: ' + name)
        need(count == record['test']['exceedances'], 'Inclusive tail count: ' + name)
        need(record['test']['assignments'] == 2**20, 'Full sign enumeration: ' + name)
        results.append({'contrast': name, 'mean': float(mean), 'raw_exact_p': float(p)})
    print(json.dumps({'status': 'PASS_PUBLIC_COUNTS_ENDPOINTS_AND_EXACT_TESTS',
        'policies': 160, 'seed_family_endpoints': 80, 'paired_differences': 60,
        'contrasts': results, 'scope': __doc__.split('\n\n', 1)[1].strip()}, indent=2))


if __name__ == '__main__':
    main()
