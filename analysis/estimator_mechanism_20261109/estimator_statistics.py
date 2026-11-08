"""Prospective ten-cluster estimator inference; independent of original Holm12.

The primitives consume source-qualified seed counts. Receipt/source validation
belongs to the separate adapter. These routines never train or select policies.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math

import numpy as np
from scipy.stats import t as student_t

ANALYSIS_ID = 'PCI-M3-ESTIMATOR-ANALYSIS-20261109-v1'
RUNTIME = {'python': '3.12.14', 'numpy': '2.3.5', 'scipy': '1.17.0'}
SEEDS = tuple(range(64001, 64011))
MODES = ('exact', 'score', 'conditional')
BASE_CANDIDATES = ('direct_02', 'direct_11')
COST_LIMIT = Fraction(1, 20)
ALPHA = Fraction(1, 20)
BOOTSTRAP_RESAMPLES = 10000
CONTRAST_IDS = ('Estimator Exact minus Score', 'Estimator Exact minus Conditional')
EXPECTED_CONTRASTS = CONTRAST_IDS
SIGN_NULL = ('Joint sign-flip invariance under the paired training-cluster '
             'sign-symmetry/exchangeability null; not a distribution-free test '
             'of mean zero alone.')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def packed(value):
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'decimal': float(value)}


def fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, dict):
        need(type(value.get('numerator')) is int and
             type(value.get('denominator')) is int and value['denominator'] > 0,
             'INVALID_PACKED_FRACTION')
        result = Fraction(value['numerator'], value['denominator'])
        if 'decimal' in value:
            need(value['decimal'] == float(result), 'PACKED_DECIMAL_MISMATCH')
        return result
    return Fraction(value)


def two_point_hypervolume(points, cost_limit=COST_LIMIT):
    """Finite two-point envelope; no interpolation or candidate reselection."""
    need(len(points) == 2, 'TWO_FROZEN_POINTS_REQUIRED')
    bound = fraction(cost_limit)
    need(bound > 0, 'POSITIVE_COST_BOUND_REQUIRED')
    data = [(fraction(c), fraction(u)) for c, u in points]
    need(all(0 <= c <= 1 and 0 <= u <= 1 for c, u in data),
         'NORMALIZED_POINT_OUT_OF_RANGE')
    boundaries = sorted({Fraction(0), bound} |
                        {c for c, _ in data if 0 <= c <= bound})
    area = Fraction(0)
    for left, right in zip(boundaries, boundaries[1:]):
        utility = max([u for c, u in data if c <= left] + [Fraction(0)])
        area += (right-left) * utility
    return area / bound


def learning_endpoints(seed_rows):
    """Validate all 60 counts and return exact endpoints plus retained points.

    Required row fields: candidate_id, training_seed, institutional_replacements,
    deliveries, joint_steps. Optional n_agents and family, when supplied, must
    equal 4 and 'direct'. No endpoint is emitted for an incomplete cohort.
    """
    need(isinstance(seed_rows, (list, tuple)), 'SEED_ROWS_REQUIRED')
    expected = {(seed, base+'_'+mode) for seed in SEEDS
                for base in BASE_CANDIDATES for mode in MODES}
    observed = {}
    for row in seed_rows:
        need(isinstance(row, dict), 'COUNT_ROW_OBJECT_REQUIRED')
        seed, candidate = row.get('training_seed'), row.get('candidate_id')
        need(type(seed) is int and (seed, candidate) in expected,
             'UNEXPECTED_ESTIMATOR_SEED_OR_CANDIDATE')
        need((seed, candidate) not in observed, 'DUPLICATE_SEED_CANDIDATE')
        need(row.get('n_agents', 4) == 4 and
             type(row.get('n_agents', 4)) is int, 'FOUR_AGENTS_REQUIRED')
        need(row.get('family', 'direct') == 'direct', 'DIRECT_FAMILY_REQUIRED')
        for name in ('institutional_replacements', 'deliveries', 'joint_steps'):
            need(type(row.get(name)) is int, 'INTEGER_COUNT_REQUIRED:'+name)
        steps = row['joint_steps']
        need(steps > 0, 'POSITIVE_JOINT_STEPS_REQUIRED')
        denominator = 4*steps
        need(0 <= row['institutional_replacements'] <= denominator and
             0 <= row['deliveries'] <= denominator, 'COUNT_OUT_OF_RANGE')
        base, mode = next((b, m) for b in BASE_CANDIDATES for m in MODES
                          if candidate == b+'_'+m)
        observed[seed, candidate] = {
            'training_seed': seed, 'candidate_id': candidate,
            'base_candidate_id': base, 'estimator': mode,
            'family': 'direct', 'n_agents': 4, 'joint_steps': steps,
            'institutional_replacements': row['institutional_replacements'],
            'deliveries': row['deliveries'],
            'cost': packed(Fraction(row['institutional_replacements'], denominator)),
            'utility': packed(Fraction(row['deliveries'], denominator)),
            'outside_primary_rectangle': Fraction(row['institutional_replacements'], denominator) > COST_LIMIT,
        }
    need(set(observed) == expected, 'COMPLETE_SIX_POLICY_TEN_CLUSTER_COHORT_REQUIRED')
    endpoints = {}
    for seed in SEEDS:
        need(len({observed[seed, base+'_'+mode]['joint_steps']
                  for base in BASE_CANDIDATES for mode in MODES}) == 1,
             'MATCHED_SIX_POLICY_EVALUATION_BUDGET_REQUIRED')
        for mode in MODES:
            rows = [observed[seed, base+'_'+mode] for base in BASE_CANDIDATES]
            need(len({r['joint_steps'] for r in rows}) == 1,
                 'MATCHED_TWO_POLICY_EVALUATION_BUDGET_REQUIRED')
            endpoints[seed, mode] = two_point_hypervolume(
                [(fraction(r['cost']), fraction(r['utility'])) for r in rows])
    return {
        'hypervolumes': endpoints,
        'policy_points': [observed[key] for key in sorted(observed)],
        'cost_limit': packed(COST_LIMIT),
    }


def _cluster_inputs(seeds, differences):
    need(tuple(seeds) == SEEDS, 'EXACT_TEN_PRESCRIBED_CLUSTER_ORDER_REQUIRED')
    need(len(differences) == len(SEEDS), 'TEN_PAIRED_DIFFERENCES_REQUIRED')
    return [fraction(x) for x in differences]


def exact_sign_flip(seeds, differences):
    differences = _cluster_inputs(seeds, differences)
    denominator = math.lcm(*(x.denominator for x in differences))
    integers = [x.numerator*(denominator//x.denominator) for x in differences]
    threshold = abs(sum(integers))
    sums = [0]
    for number in integers:
        sums = [old+number for old in sums] + [old-number for old in sums]
    exceedances = sum(abs(value) >= threshold for value in sums)
    return {
        'statistic': 'absolute_mean_paired_training_cluster_difference',
        'statistic_abs_mean': packed(abs(sum(differences)/len(differences))),
        'assignments': 1 << len(SEEDS), 'exceedances': exceedances,
        'p_value': packed(Fraction(exceedances, 1 << len(SEEDS))),
        'tail': 'inclusive_exact_absolute_statistic',
        'monte_carlo_correction': False, 'null_assumption': SIGN_NULL,
    }


def bootstrap_seed(contrast_id):
    need(contrast_id in CONTRAST_IDS, 'UNKNOWN_ESTIMATOR_CONTRAST')
    material = ANALYSIS_ID+'/'+contrast_id
    return int.from_bytes(hashlib.sha256(material.encode('utf-8')).digest()[:8], 'big')


def _linear_quantile(sorted_integers, probability):
    location = Fraction(probability)*(len(sorted_integers)-1)
    lower = location.numerator//location.denominator
    weight = location-lower
    upper = min(lower+1, len(sorted_integers)-1)
    return Fraction(sorted_integers[lower])*(1-weight) + Fraction(sorted_integers[upper])*weight


def paired_bootstrap(contrast_id, seeds, differences):
    differences = _cluster_inputs(seeds, differences)
    seed = bootstrap_seed(contrast_id)
    indices = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(SEEDS), size=(BOOTSTRAP_RESAMPLES, len(SEEDS)), dtype=np.int64)
    denominator = math.lcm(*(x.denominator for x in differences))
    values = [x.numerator*(denominator//x.denominator) for x in differences]
    sums = sorted(sum(values[int(i)] for i in row) for row in indices)
    mean_denominator = denominator*len(SEEDS)
    lower = _linear_quantile(sums, Fraction(1,40))/mean_denominator
    upper = _linear_quantile(sums, Fraction(39,40))/mean_denominator
    return {
        'method': 'paired_whole_training_cluster_percentile',
        'confidence': .95, 'pointwise': True, 'simultaneous': False,
        'resamples': BOOTSTRAP_RESAMPLES, 'rng': 'NumPy Generator(PCG64)',
        'seed': seed, 'seed_derivation_utf8': ANALYSIS_ID+'/'+contrast_id,
        'quantile_method': 'linear', 'numpy_version': np.__version__,
        'endpoint_arithmetic': 'exact_rational_linear_quantiles_of_integer_scaled_resample_sums',
        'resample_index_sha256': hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest(),
        'lower': packed(lower), 'upper': packed(upper),
    }


def contrast_statistics(contrast_id, seeds, differences):
    need(contrast_id in CONTRAST_IDS, 'UNKNOWN_ESTIMATOR_CONTRAST')
    values = _cluster_inputs(seeds, differences)
    count = len(values)
    mean = sum(values)/count
    variance = sum((x-mean)**2 for x in values)/(count-1)
    sd = math.sqrt(float(variance))
    test = exact_sign_flip(seeds, values)
    critical = float(student_t.ppf(.975, count-1))
    if variance == 0:
        dz = {'status': 'UNDEFINED_ZERO_SD', 'value': None}
        t95 = {'status': 'UNDEFINED_ZERO_SD', 'lower': None, 'upper': None}
    else:
        radius = critical*sd/math.sqrt(count)
        dz = {'status': 'DEFINED', 'value': float(mean)/sd}
        t95 = {'status': 'DEFINED', 'lower': float(mean)-radius,
               'upper': float(mean)+radius}
    t95.update(method='paired_t', confidence=.95, df=count-1,
               critical_value=critical, pointwise=True, simultaneous=False,
               descriptive_sensitivity=True)
    return {
        'contrast_id': contrast_id, 'training_clusters': count,
        'mean_effect': packed(mean), 'paired_sample_variance': packed(variance),
        'paired_sample_sd': sd, 'p_value': test['p_value'], 'test': test,
        'bootstrap95': paired_bootstrap(contrast_id, seeds, values),
        'paired_t95': t95, 'dz': dz,
        'cluster_differences': [{'training_seed': seed, 'difference': packed(value)}
                                for seed, value in zip(seeds, values)],
    }


def holm2(records, require_complete=False):
    need(isinstance(records, (list, tuple)), 'CONTRAST_RECORDS_REQUIRED')
    need(all(isinstance(record, dict) for record in records), 'CONTRAST_RECORD_OBJECT_REQUIRED')
    ids = [r.get('contrast_id') for r in records]
    need(all(identifier in CONTRAST_IDS for identifier in ids), 'UNKNOWN_ESTIMATOR_CONTRAST')
    need(len(ids) == len(set(ids)), 'DUPLICATE_ESTIMATOR_CONTRAST')
    need(all('adjusted_p_value' not in r and 'reject' not in r for r in records),
         'RAW_UNADJUSTED_RECORDS_REQUIRED')
    for record in records:
        need(0 <= fraction(record['p_value']) <= 1, 'P_VALUE_OUT_OF_RANGE')
    pending = [identifier for identifier in CONTRAST_IDS if identifier not in ids]
    complete = not pending
    need(not require_complete or complete, 'COMPLETE_SEPARATE_ESTIMATOR_HOLM2_REQUIRED')
    ordered = sorted((dict(r) for r in records),
                     key=lambda r: CONTRAST_IDS.index(r['contrast_id']))
    if complete:
        rank_order = sorted(ordered, key=lambda r: (fraction(r['p_value']),
                                                  CONTRAST_IDS.index(r['contrast_id'])))
        previous = Fraction(0)
        for rank, record in enumerate(rank_order):
            adjusted = max(previous, min(Fraction(1), (2-rank)*fraction(record['p_value'])))
            record['adjusted_p_value'] = packed(adjusted)
            record['reject'] = adjusted <= ALPHA
            previous = adjusted
    return {
        'status': 'COMPLETE_SEPARATE_ESTIMATOR_FAMILY' if complete else 'DESCRIPTIVE_PARTIAL_ESTIMATOR_FAMILY',
        'family_id': 'SEPARATE_ESTIMATOR_HOLM2',
        'family_size': 2, 'family_complete': complete, 'alpha': packed(ALPHA),
        'pending_contrasts': pending, 'contrasts': ordered,
        'primary_Holm12_changed': False,
    }


def analyze_rows(seed_rows):
    data = learning_endpoints(seed_rows)
    endpoints = data['hypervolumes']
    contrasts = []
    for identifier, mode in zip(CONTRAST_IDS, ('score', 'conditional')):
        differences = [endpoints[seed, 'exact']-endpoints[seed, mode] for seed in SEEDS]
        contrasts.append(contrast_statistics(identifier, SEEDS, differences))
    return {
        'analysis_id': ANALYSIS_ID, **holm2(contrasts, require_complete=True),
        'endpoint_data': {
            'cost_limit': data['cost_limit'],
            'policy_points': data['policy_points'],
            'estimator_cluster_hypervolumes': [
                {'training_seed': seed, 'estimator': mode,
                 'hypervolume': packed(endpoints[seed, mode])}
                for seed in SEEDS for mode in MODES],
        },
    }
