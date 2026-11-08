"""Artificial complete-shaped adapter controls; never scientific outcomes."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
from fractions import Fraction

import estimator_receipts as lr
import estimator_adapter_fixtures as fixtures
import estimator_statistics as stats

ROOT = Path(__file__).resolve().parent


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refused(call, expected):
    try:
        call()
    except (ValueError, FileNotFoundError) as error:
        assert expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError('Expected refusal: ' + expected)


def run():
    controls = []
    with tempfile.TemporaryDirectory(prefix='pci_estimator_adapter_synthetic_') as temporary:
        work = Path(temporary)
        cohort, selection, ledger = fixtures.make_fixture(work, 'warehouse')
        invoke = lambda: lr.load_learning_receipts(cohort, fixtures.SOURCE, selection, ledger,
                                                   _test_only_synthetic=True)
        result = invoke()
        assert len(result['seed_rows']) == 60 and len(result['profile_rows']) == 240
        assert sum(r['joint_steps'] for r in result['seed_rows']) == 1228800
        packet = stats.analyze_rows(result['seed_rows'])
        assert packet['family_complete'] and len(packet['contrasts']) == 2
        assert result['provenance']['synthetic_validation_only']
        controls.append({'control': 'Complete synthetic 60-policy / 4,800-summary closure and endpoint integration', 'status': 'PASS'})
        actual = {(row['training_seed'], row['estimator']):
                  Fraction(row['hypervolume']['numerator'], row['hypervolume']['denominator'])
                  for row in packet['endpoint_data']['estimator_cluster_hypervolumes']}
        expected = {}
        for seed in range(64001, 64011):
            for mode, offset in [('exact',0), ('score',1), ('conditional',2)]:
                j = seed - 64001
                c1, c2 = Fraction(2+offset+j%2,1024), Fraction(1+offset+j%2,1024)
                u1, u2 = Fraction(9-offset+j%3,1024), Fraction(8-offset+j%3,1024)
                bound = Fraction(1,20)
                # Independent two-rectangle inclusion-exclusion formula.
                expected[seed,mode] = ((bound-c1)*u1 + (bound-c2)*u2
                                      - (bound-max(c1,c2))*min(u1,u2))/bound
        assert actual == expected
        for contrast, mode in zip(packet['contrasts'], ['score','conditional']):
            mean = sum((expected[seed,'exact']-expected[seed,mode]
                        for seed in range(64001,64011)), Fraction(0))/10
            assert mean > 0
            assert Fraction(contrast['mean_effect']['numerator'], contrast['mean_effect']['denominator']) == mean
        controls.append({'control': 'Unequal mode/seed full fixture reproduces independent nonzero endpoints and contrast wiring', 'status': 'PASS'})

        def record_refusal(label, call, expected):
            controls.append({'control': label, 'status': 'PASS_REFUSED', 'error': refused(call, expected)})

        def mutate(path, change, label, expected):
            original = path.read_bytes()
            try:
                put(path, change(json.loads(original)))
                record_refusal(label, invoke, expected)
            finally:
                path.write_bytes(original)

        record_refusal('Artificial marker is refused in production mode',
            lambda: lr.load_learning_receipts(cohort, fixtures.SOURCE, selection, ledger),
            'SYNTHETIC_COHORT_REFUSED')
        final = cohort / 'STUDY_COMPLETE.json'
        inputs = ledger.parent.parent
        evaluation = cohort / 'evaluation'
        mutate(final, lambda x: dict(x, stage='confirmation'), 'Primary stage cannot become estimator stage', 'COMPLETE_ESTIMATOR_COHORT_REQUIRED')
        mutate(final, lambda x: dict(x, trained_policies=59), 'Incomplete training cohort refused', 'PRESCRIBED_TRAINING_BUDGET')
        mutate(final, lambda x: dict(x, all_owned_worker_processes_closed=False), 'Live owned worker refused', 'OWNED_WORKERS_NOT_CLOSED')
        mutate(inputs / 'SOURCE_DESIGN_LOCK.json', lambda x: dict(x, stage='confirmation'), 'Explicit stage authority required', 'EXPLICIT_ESTIMATOR_DESIGN_SCOPE_REQUIRED')
        mutate(inputs / 'SOURCE_DESIGN_LOCK.json', lambda x: dict(x, estimator_analysis_recipe_sha256='0'*64), 'Prospective recipe binding required', 'PROSPECTIVE_ESTIMATOR_ANALYSIS_RECIPE_BINDING_REQUIRED')
        mutate(inputs / 'SOURCE_DESIGN_LOCK.json', lambda x: dict(x, estimator_analysis_implementation_lock_sha256='0'*64), 'Prospective implementation binding required', 'PROSPECTIVE_ESTIMATOR_ANALYSIS_IMPLEMENTATION_BINDING_REQUIRED')
        mutate(selection, lambda x: dict(x, estimator_ablation_candidate_ids=['direct_01','direct_02']), 'Selection body cannot be edited', 'PROSPECTIVELY_BOUND_DEVELOPMENT_SELECTION_REQUIRED')
        mutate(inputs / 'CUDA_READINESS.json', lambda x: dict(x, estimator_modes_pass={'exact': True}), 'All three actual estimator modes required', 'ALL_ESTIMATOR_MODES_READINESS_REQUIRED')
        config = ledger.parent / json.loads(ledger.read_text())['cases'][0]['path']
        mutate(config, lambda x: dict(x, risk_samples=8), 'Unbound training configuration refused', 'TRAINING_CONFIG_HASH')
        missing = next((evaluation / 'cases').glob('*.json'))
        original = missing.read_bytes()
        missing.unlink()
        try:
            record_refusal('A missing scheduled case cannot be ignored', invoke, 'CASE_FILE_MEMBERSHIP')
        finally:
            missing.write_bytes(original)
        extra = evaluation / 'cases/UNSCHEDULED.json'
        extra.write_text('{}')
        try:
            record_refusal('Unexpected case file refused', invoke, 'CASE_FILE_MEMBERSHIP')
        finally:
            extra.unlink()
        mutate(missing, lambda x: dict(x, totals=dict(x['totals'], deliveries=3)), 'Unbound summary change refused', 'CASE_RECEIPT_HASH')
        # Rebind modified summary receipt to verify semantic bounds independently
        # of the first raw-hash gate. These are still artificial local records.
        complete = evaluation / 'EVALUATION_COMPLETE.json'
        old_complete, old_case = complete.read_bytes(), missing.read_bytes()
        try:
            body = json.loads(old_case)
            body['totals']['deliveries'] = 1025
            put(missing, body)
            value = json.loads(old_complete)
            value['case_receipts'][body['case']['case_id']] = sha(missing)
            put(complete, value)
            record_refusal('Hash-rebound impossible count refused', invoke, 'COUNT_EXPOSURE_OR_HAMMING_BOUND')
        finally:
            missing.write_bytes(old_case)
            complete.write_bytes(old_complete)
        # Rebind analysis hashes so aggregate-duplication checks are reached.
        aggregate = evaluation / 'ANALYSIS.json'
        originals = {p: p.read_bytes() for p in [aggregate, complete, final]}
        try:
            value = json.loads(originals[aggregate])
            value['equal_profile_seed_rows'][1] = copy.deepcopy(value['equal_profile_seed_rows'][0])
            put(aggregate, value)
            for path in [complete, final]:
                value = json.loads(originals[path])
                value['analysis_sha256'] = sha(aggregate)
                put(path, value)
            record_refusal('Hash-rebound duplicate aggregate cluster refused', invoke, 'INCOMPLETE_OR_DUPLICATE:ANALYSIS_SEED_ROWS')
        finally:
            for path, body in originals.items():
                path.write_bytes(body)
        # Raw duplicate JSON keys cannot be masked by schema checks.
        original = final.read_bytes()
        try:
            final.write_bytes(original[:-2] + b',"stage":"estimator_ablation"}\n')
            record_refusal('Duplicate producer JSON key refused', invoke, 'DUPLICATE_JSON_KEY:stage')
        finally:
            final.write_bytes(original)
        target = work / 'completion_backup.json'
        target.write_bytes(original)
        final.unlink()
        final.symlink_to(target)
        try:
            record_refusal('Symlink completion input refused', invoke, 'SYMLINK_INPUT')
        finally:
            final.unlink()
            final.write_bytes(original)
        path = work / 'held_body.txt'
        path.write_text('body')
        snapshot = lr.Snapshot()
        snapshot.body(path)
        path.write_text('changed')
        record_refusal('Held inputs are rechecked after use', snapshot.check, 'INPUT_CHANGED_AFTER_READ')
        empty = work / 'empty_source'
        empty.mkdir()
        record_refusal('Wrong executable source map refused',
            lambda: lr.load_learning_receipts(cohort, empty, selection, ledger, _test_only_synthetic=True),
            'UNQUALIFIED_SOURCE_MAP')
        # Restored fixture still passes after negative tests; no mutation leak.
        assert len(invoke()['seed_rows']) == 60
        controls.append({'control': 'Restored synthetic fixture passes after all negative controls', 'status': 'PASS'})
    receipt = {'schema': 'PCI-ESTIMATOR-RECEIPT-ADAPTER-TESTS-v1', 'status': 'PASS',
               'scope': 'Artificial complete-shaped software controls only; no scientific outcomes or raw trace/model replay.',
               'controls': controls,
               'sources': {n: sha(ROOT/n) for n in ['estimator_receipts.py','estimator_adapter_fixtures.py','test_estimator_receipts.py']}}
    output = ROOT / 'ESTIMATOR_ADAPTER_TEST_RESULTS.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'controls': len(controls), 'out': str(output)}))


if __name__ == '__main__':
    run()
