#!/usr/bin/env python3
"""Run only synthetic software controls and emit their implementation lock.

This does not authorize training. A root-reviewed scientific source/design and
actual-runtime readiness lock remain separate. Run before opening scientific
outcomes. Original primary analysis modules are never edited or imported.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--receipt-tag', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args = parser.parse_args()
    import numpy as np
    import scipy
    import estimator_analysis as analysis
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__}
    if versions != analysis.RUNTIME_VERSIONS:
        raise RuntimeError('Qualification requires the declared exact analysis runtime.')
    analysis.verify_recipe()
    before = {name: sha(ROOT/name) for name in analysis.RUNTIME_FILES}
    lock_path = ROOT/'ESTIMATOR_ANALYSIS_IMPLEMENTATION_LOCK.json'
    lock_path.write_text(json.dumps({'schema': 'PCI-ESTIMATOR-ANALYSIS-IMPLEMENTATION-LOCK-v1',
                                    'status': 'QUALIFICATION_IN_PROGRESS_NOT_LAUNCH'}, indent=2)+'\n')
    stats_receipt = ROOT / ('ESTIMATOR_STATISTICS_TEST_RESULTS_'+args.receipt_tag+'.json')
    subprocess.run([sys.executable, '-B', str(ROOT/'test_estimator_statistics.py'), '--out', str(stats_receipt)], check=True)
    subprocess.run([sys.executable, '-B', str(ROOT/'test_estimator_receipts.py'), str(args.source_root.resolve())], check=True)
    adapter_receipt = ROOT/'ESTIMATOR_ADAPTER_TEST_RESULTS.json'
    control_receipts = [stats_receipt, adapter_receipt]
    lock = {'schema': 'PCI-ESTIMATOR-ANALYSIS-IMPLEMENTATION-LOCK-v1',
            'status': 'PASS_SYNTHETIC_QUALIFICATION', 'recipe_sha256': analysis.RECIPE_SHA256,
            'runtime_sources': before, 'analysis_runtime_versions': versions,
            'qualification_receipts': {p.name: sha(p) for p in control_receipts},
            'scope': 'Prospective synthetic numerical, complete-shaped receipt and pipeline qualification. No scientific launch authority or external attestation.',
            'independent_review_required_before_scientific_launch': True}
    # Exercise the candidate production gates in an isolated copy. The canonical
    # lock remains non-launchable if a process is killed or any pipeline test fails.
    with tempfile.TemporaryDirectory(prefix='pci_estimator_candidate_analysis_') as temporary:
        candidate = Path(temporary)
        for name in [*analysis.RUNTIME_FILES, 'ESTIMATOR_ANALYSIS_RECIPE.json',
                     'test_estimator_pipeline.py', 'estimator_adapter_fixtures.py']:
            shutil.copyfile(ROOT/name, candidate/name)
        (candidate/'ESTIMATOR_ANALYSIS_IMPLEMENTATION_LOCK.json').write_text(
            json.dumps(lock, sort_keys=True, indent=2)+'\n')
        subprocess.run([sys.executable, '-B', str(candidate/'test_estimator_pipeline.py'),
                        str(args.source_root.resolve())], check=True)
        shutil.copyfile(candidate/'ESTIMATOR_PIPELINE_TEST_RESULTS.json',
                        ROOT/'ESTIMATOR_PIPELINE_TEST_RESULTS.json')
    pipeline_receipt = ROOT/'ESTIMATOR_PIPELINE_TEST_RESULTS.json'
    control_receipts.append(pipeline_receipt)
    if before != {name: sha(ROOT/name) for name in analysis.RUNTIME_FILES}:
        raise RuntimeError('Executable analysis source changed during qualification.')
    lock['qualification_receipts'] = {p.name: sha(p) for p in control_receipts}
    lock['qualification_sources'] = {p.name: sha(p) for p in [ROOT/'test_estimator_statistics.py', ROOT/'test_estimator_receipts.py', ROOT/'test_estimator_pipeline.py', ROOT/'estimator_adapter_fixtures.py', Path(__file__)]}
    lock['controls'] = {'statistics': json.loads(stats_receipt.read_bytes())['tests_run'],
                        'receipt_adapter': len(json.loads(adapter_receipt.read_bytes())['controls']),
                        'pipeline': len(json.loads(pipeline_receipt.read_bytes())['controls'])}
    lock_path.write_text(json.dumps(lock, sort_keys=True, indent=2)+'\n')
    analysis.verify_implementation_lock()
    print(json.dumps({'status': lock['status'], 'recipe_sha256': analysis.RECIPE_SHA256,
                      'implementation_lock_sha256': sha(lock_path), 'controls': lock['controls']}))


if __name__ == '__main__':
    main()
