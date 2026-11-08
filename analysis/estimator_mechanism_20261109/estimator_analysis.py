#!/usr/bin/env python3
"""Prospective estimator-cohort analysis, separate from the original Holm12.

The CLI accepts only a complete source-bound warehouse estimator cohort.
It does not replay absent raw models/traces, attest a publisher, or access
remote services. All output files are immutable fresh paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import estimator_statistics as stats
from estimator_receipts import load_learning_receipts, need, canonical

RECIPE_ID = 'PCI-M3-ESTIMATOR-ANALYSIS-20261109-v1'
RECIPE_SHA256 = '1c70ed003749fd69487bdfa1fef8ddff781cb0f5f87b3f8e59d06d9bd5c2053d'
RUNTIME_FILES = ('estimator_analysis.py', 'estimator_statistics.py', 'estimator_receipts.py')
RUNTIME_VERSIONS = {'python': '3.12.14', 'numpy': '2.3.5', 'scipy': '1.17.0'}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_recipe():
    path = Path(__file__).with_name('ESTIMATOR_ANALYSIS_RECIPE.json')
    need(sha_file(path) == RECIPE_SHA256, 'ESTIMATOR_ANALYSIS_RECIPE_CHANGED')
    recipe = json.loads(path.read_bytes())
    need(recipe['recipe_id'] == RECIPE_ID and recipe['training_seeds'] == list(stats.SEEDS),
         'ESTIMATOR_RECIPE_IDENTITY')
    need(recipe['canonical_contrast_ids'] == list(stats.CONTRAST_IDS), 'ESTIMATOR_CONTRAST_IDENTITY')
    return recipe


def verify_implementation_lock():
    import numpy as np
    import scipy
    root = Path(__file__).resolve().parent
    lock = json.loads((root / 'ESTIMATOR_ANALYSIS_IMPLEMENTATION_LOCK.json').read_bytes())
    need(lock.get('schema') == 'PCI-ESTIMATOR-ANALYSIS-IMPLEMENTATION-LOCK-v1'
         and lock.get('status') == 'PASS_SYNTHETIC_QUALIFICATION',
         'ESTIMATOR_ANALYSIS_QUALIFICATION_REQUIRED')
    need(lock.get('recipe_sha256') == RECIPE_SHA256, 'ESTIMATOR_RECIPE_LOCK_MISMATCH')
    actual = {name: sha_file(root / name) for name in RUNTIME_FILES}
    need(lock.get('runtime_sources') == actual, 'ESTIMATOR_ANALYSIS_SOURCE_CHANGED')
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__}
    need(versions == RUNTIME_VERSIONS == lock.get('analysis_runtime_versions'),
         'ESTIMATOR_ANALYSIS_RUNTIME_REQUALIFICATION_REQUIRED')
    return {'implementation_lock_sha256': sha_file(root / 'ESTIMATOR_ANALYSIS_IMPLEMENTATION_LOCK.json'),
            'runtime_sources': actual, 'analysis_runtime_versions': versions}


def learning_packet(cohort, source_root, selection_path, case_manifest_path,
                    *, _test_only_synthetic=False):
    verify_recipe()
    implementation = verify_implementation_lock()
    inputs = {name: str(Path(value).resolve()) for name, value in
              [('cohort', cohort), ('source_root', source_root),
               ('selection_path', selection_path), ('case_manifest_path', case_manifest_path)]}
    data = load_learning_receipts(**inputs, _test_only_synthetic=_test_only_synthetic)
    result = stats.analyze_rows(data['seed_rows'])
    result.update(schema='PCI-ESTIMATOR-INFERENCE-RESULT-v1', recipe_id=RECIPE_ID,
                  recipe_sha256=RECIPE_SHA256, implementation=implementation,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  original_global_Holm12_changed=False,
                  provenance={'kind': 'SOURCE_BOUND_COMPLETE_ESTIMATOR_LEARNING_COHORT',
                              'inputs': inputs, 'receipt_adapter': data['provenance']})
    if data['provenance']['synthetic_validation_only']:
        result['scientific_result'] = False
        result['qualification_scope'] = 'Artificial complete-shaped software fixture only.'
    else:
        result['scientific_result'] = True
    return result


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical(value) + b'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--case-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = learning_packet(args.cohort, args.source_root, args.selection, args.case_manifest)
    write_once(args.out, result)
    print(json.dumps({'status': result['status'], 'out': str(args.out),
                      'contrast_count': len(result['contrasts'])}))


if __name__ == '__main__':
    main()
