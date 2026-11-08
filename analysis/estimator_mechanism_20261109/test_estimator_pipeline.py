"""Qualified synthetic pipeline controls; no real estimator cohort is opened."""
from pathlib import Path
import copy
import hashlib
import json
import subprocess
import sys
import tempfile

import estimator_adapter_fixtures as fixtures
import estimator_analysis as analysis

ROOT = Path(__file__).resolve().parent


def refused(call, expected):
    try:
        call()
    except (ValueError, FileExistsError) as error:
        assert expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError('Expected refusal: ' + expected)


def run():
    controls = []
    assert analysis.verify_recipe()['recipe_id'] == analysis.RECIPE_ID
    implementation = analysis.verify_implementation_lock()
    controls.append({'control': 'Qualified recipe, runtime and executable source hashes match', 'status': 'PASS'})
    with tempfile.TemporaryDirectory(prefix='pci_estimator_pipeline_synthetic_') as temporary:
        cohort, selection, ledger = fixtures.make_fixture(Path(temporary), 'warehouse')
        result = analysis.learning_packet(cohort, fixtures.SOURCE, selection, ledger, _test_only_synthetic=True)
        assert result['scientific_result'] is False
        assert result['family_complete'] is True and len(result['contrasts']) == 2
        assert all(row['mean_effect']['decimal'] > 0 for row in result['contrasts'])
        assert result['original_global_Holm12_changed'] is False
        assert len(result['endpoint_data']['policy_points']) == 60
        json.dumps(result, allow_nan=False)
        controls.append({'control': 'Full unequal 60-policy synthetic pipeline is explicitly non-scientific', 'status': 'PASS'})
        destination = Path(temporary) / 'output.json'
        command = [sys.executable, '-B', str(ROOT/'estimator_analysis.py'), '--cohort', str(cohort),
                   '--source-root', str(fixtures.SOURCE), '--selection', str(selection),
                   '--case-manifest', str(ledger), '--out', str(destination)]
        process = subprocess.run(command, capture_output=True, text=True)
        assert process.returncode != 0 and 'SYNTHETIC_COHORT_REFUSED' in process.stderr
        assert not destination.exists()
        controls.append({'control': 'Production CLI refuses the artificial marker and writes no output', 'status': 'PASS_REFUSED'})
        analysis.write_once(destination, result)
        assert json.loads(destination.read_bytes()) == result
        refused(lambda: analysis.write_once(destination, result), 'File exists')
        controls.append({'control': 'A result destination cannot be overwritten', 'status': 'PASS_REFUSED'})
    original_hash = analysis.RECIPE_SHA256
    try:
        analysis.RECIPE_SHA256 = '0'*64
        refused(analysis.verify_recipe, 'ESTIMATOR_ANALYSIS_RECIPE_CHANGED')
    finally:
        analysis.RECIPE_SHA256 = original_hash
    controls.append({'control': 'A changed recipe identity fails before inference', 'status': 'PASS_REFUSED'})
    versions = copy.deepcopy(analysis.RUNTIME_VERSIONS)
    try:
        analysis.RUNTIME_VERSIONS = dict(versions, numpy='WRONG_VERSION')
        refused(analysis.verify_implementation_lock, 'ESTIMATOR_ANALYSIS_RUNTIME_REQUALIFICATION_REQUIRED')
    finally:
        analysis.RUNTIME_VERSIONS = versions
    controls.append({'control': 'Runtime changes require requalification', 'status': 'PASS_REFUSED'})
    output = {'schema': 'PCI-ESTIMATOR-PIPELINE-TESTS-v1', 'status': 'PASS',
              'scope': 'Synthetic software controls only. No scientific results, raw replay, or remote job.',
              'controls': controls,
              'runtime_sources': implementation['runtime_sources'],
              'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'ESTIMATOR_PIPELINE_TEST_RESULTS.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'controls': len(controls)}))


if __name__ == '__main__':
    run()
