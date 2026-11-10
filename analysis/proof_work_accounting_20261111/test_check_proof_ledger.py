"""Synthetic contract/byte-binding controls; no real cryptographic execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check_proof_ledger import check_ledger, LedgerError


VKEY = '1' * 64
AUDITOR = '2' * 64


def fixture(root):
    root = Path(root)
    calls = []; records = []
    def artifact(name, value):
        body = (json.dumps(value, sort_keys=True) + '\n').encode()
        (root / name).write_bytes(body)
        return {'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    for i in range(5):
        row = {'call_id': f'call-{i}', 'offered_id': f'offer-{i}',
               'attempt_id': f'attempt-{i}', 'status': 'VALID',
               'return_artifact': artifact(f'return-{i}.json', {'synthetic': True, 'call': i})}
        if i != 4:
            row['prepared_artifact'] = artifact(f'prepared-{i}.json', {'synthetic': True, 'attempt': i})
        else:
            row['host_preparation_status'] = 'FAILED_AFTER_VALID_PROOF'
        calls.append(row)
        records.append({'call_id': row['call_id'], 'attempt_id': row['attempt_id'],
            'return_artifact_sha256': row['return_artifact']['sha256'], 'valid': True,
            'native_binding_valid': True,
            'prepared_artifact_sha256': row.get('prepared_artifact', {}).get('sha256')})
    apps = [{'attempt_id': f'attempt-{i}', 'offered_id': f'offer-{i}',
             'status': 'APPLIED', 'domain': 100, 'seq_before': i,
             'seq_after': i + 1, 'task_units': 2} for i in range(3)]
    apps.extend([{'attempt_id': 'attempt-2', 'offered_id': 'offer-2',
                 'status': 'DUPLICATE', 'domain': 100, 'seq_before': 3, 'seq_after': 3,
                 'new_commit': False, 'new_effect': False},
                {'attempt_id': 'attempt-3', 'offered_id': 'offer-3',
                 'status': 'REJECTED', 'code': 'STALE_PREDECESSOR_OR_VERSION'}])
    ledger = {'schema': 'PCI-PROOF-WORK-LEDGER-v1', 'complete_drain': True,
              'offers': [{'offered_id': f'offer-{i}'} for i in range(6)],
              'proof_calls': calls, 'applications': apps,
              'preproof_rejections': [{'offered_id': 'offer-5', 'proof_called': False}]}
    audit = {'schema': 'PCI-INDEPENDENT-PROOF-CORPUS-AUDIT-v1',
             'status': 'PASS_ACTUAL_VKEY_NATIVE_REPLAY', 'vkey_sha256': VKEY,
             'source_identity': {'auditor_source_sha256': AUDITOR},
             'synthetic_fixture': True, 'records': records}
    return ledger, audit


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger, self.audit = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def check(self, ledger=None, audit=None, **kw):
        return check_ledger(ledger or self.ledger, audit or self.audit,
            corpus_root=self.root, expected_vkey_sha256=VKEY,
            expected_auditor_sha256=AUDITOR, **kw)

    def rejects(self, change):
        l, a = copy.deepcopy(self.ledger), copy.deepcopy(self.audit)
        change(l, a)
        with self.assertRaises((LedgerError, OSError)):
            self.check(l, a)

    def test_positive_retained_byte_contract(self):
        r = self.check()
        self.assertEqual((r['proof_rpc_invocations'], r['valid_generated_proofs'],
            r['distinct_durably_applied_proof_attempts'], r['valid_unapplied_proofs'],
            r['valid_stale_unapplied_proofs'], r['preproof_rejections'],
            r['duplicate_retries'], r['task_units']), (5, 5, 3, 2, 1, 1, 1, 6))
        self.assertFalse(r['scientific_result'])
        self.assertFalse(r['proof_verifier_executed_by_this_module'])

    def test_valid_return_without_prepared_body_is_retained_unused(self):
        r = self.check()
        self.assertIn('attempt-4', r['valid_unapplied_attempt_ids'])

    def test_failed_and_pending_rpc_partition(self):
        l = copy.deepcopy(self.ledger)
        l['offers'] += [{'offered_id': 'failed'}, {'offered_id': 'pending'}]
        l['proof_calls'] += [
            {'call_id': 'failed', 'offered_id': 'failed', 'attempt_id': 'failed', 'status': 'FAILED_NO_VALID'},
            {'call_id': 'pending', 'offered_id': 'pending', 'attempt_id': 'pending', 'status': 'INFLIGHT'}]
        l['complete_drain'] = False
        r = self.check(l)
        self.assertEqual((r['proof_rpc_invocations'], r['failed_before_valid'], r['inflight']), (7, 1, 1))

    def test_old_attempt_retry_after_later_commits(self):
        l = copy.deepcopy(self.ledger)
        l['applications'][3].update(attempt_id='attempt-0', offered_id='offer-0', ack_sequence=1)
        self.assertEqual(self.check(l)['duplicate_retries'], 1)

    def test_retry_ack_identity_rejected(self):
        self.rejects(lambda l,a: l['applications'][3].update(ack_sequence=99))

    def test_complete_drain_rejects_pending_call(self):
        l, a = copy.deepcopy(self.ledger), copy.deepcopy(self.audit)
        l['proof_calls'][4].update(status='INFLIGHT')
        l['proof_calls'][4].pop('return_artifact')
        a['records'].pop()
        with self.assertRaises(LedgerError): self.check(l, a)

    def test_preproof_reject_cannot_have_call(self):
        self.rejects(lambda l,a: l['preproof_rejections'][0].update(offered_id='offer-0'))

    def test_prefilter_cannot_claim_proof_called(self):
        self.rejects(lambda l,a: l['preproof_rejections'][0].update(proof_called=True))

    def test_duplicate_call_rejected(self):
        self.rejects(lambda l,a: l['proof_calls'].append(copy.deepcopy(l['proof_calls'][0])))

    def test_duplicate_attempt_rejected(self):
        self.rejects(lambda l,a: l['proof_calls'][1].update(attempt_id='attempt-0'))

    def test_unknown_offer_rejected(self):
        self.rejects(lambda l,a: l['proof_calls'][0].update(offered_id='unknown'))

    def test_changed_return_bytes_rejected(self):
        (self.root/'return-0.json').write_bytes(b'changed')
        with self.assertRaises(LedgerError): self.check()

    def test_symlink_return_rejected(self):
        body = (self.root/'return-0.json').read_bytes()
        (self.root/'actual.json').write_bytes(body)
        (self.root/'return-0.json').unlink()
        (self.root/'return-0.json').symlink_to(self.root/'actual.json')
        with self.assertRaises(LedgerError): self.check()

    def test_parent_escape_rejected(self):
        self.rejects(lambda l,a: l['proof_calls'][0]['return_artifact'].update(path='../escape'))

    def test_missing_valid_return_rejected(self):
        self.rejects(lambda l,a: l['proof_calls'][0].pop('return_artifact'))

    def test_missing_audit_rejected(self):
        self.rejects(lambda l,a: a['records'].pop())

    def test_extra_audit_rejected(self):
        self.rejects(lambda l,a: a['records'].append(dict(a['records'][0], call_id='unknown')))

    def test_duplicate_audit_rejected(self):
        self.rejects(lambda l,a: a['records'].append(copy.deepcopy(a['records'][0])))

    def test_audit_wrong_return_hash_rejected(self):
        self.rejects(lambda l,a: a['records'][0].update(return_artifact_sha256='3'*64))

    def test_audit_wrong_prepared_hash_rejected(self):
        self.rejects(lambda l,a: a['records'][0].update(prepared_artifact_sha256='3'*64))

    def test_audit_wrong_vkey_rejected(self):
        self.rejects(lambda l,a: a.update(vkey_sha256='3'*64))

    def test_audit_wrong_source_rejected(self):
        self.rejects(lambda l,a: a['source_identity'].update(auditor_source_sha256='3'*64))

    def test_valid_boolean_without_native_binding_rejected(self):
        self.rejects(lambda l,a: a['records'][0].update(native_binding_valid=False))

    def test_claimed_valid_failed_actual_audit_rejected(self):
        self.rejects(lambda l,a: a['records'][0].update(valid=False))

    def test_stale_proof_without_audit_rejected(self):
        self.rejects(lambda l,a: a['records'].__delitem__(3))

    def test_valid_proof_misclassified_failed_rejected(self):
        self.rejects(lambda l,a: l['proof_calls'][4].update(status='FAILED_NO_VALID'))

    def test_applied_twice_rejected(self):
        self.rejects(lambda l,a: l['applications'].append(copy.deepcopy(l['applications'][0])))

    def test_applied_sequence_gap_rejected(self):
        self.rejects(lambda l,a: l['applications'][0].update(seq_after=2))

    def test_duplicate_changes_commit_prefix_rejected(self):
        self.rejects(lambda l,a: l['applications'][3].update(seq_after=4))

    def test_duplicate_adds_effect_rejected(self):
        self.rejects(lambda l,a: l['applications'][3].update(new_effect=True))

    def test_duplicate_adds_commit_rejected(self):
        self.rejects(lambda l,a: l['applications'][3].update(new_commit=True))

    def test_duplicate_without_original_application_rejected(self):
        self.rejects(lambda l,a: l['applications'].__delitem__(2))

    def test_unknown_application_rejected(self):
        self.rejects(lambda l,a: l['applications'][0].update(attempt_id='unknown'))

    def test_application_of_unprepared_valid_proof_rejected(self):
        self.rejects(lambda l,a: l['applications'][0].update(attempt_id='attempt-4', offered_id='offer-4'))

    def test_stale_and_applied_partition_rejected(self):
        self.rejects(lambda l,a: l['applications'][-1].update(attempt_id='attempt-0', offered_id='offer-0'))

    def test_unprepared_valid_status_required(self):
        self.rejects(lambda l,a: l['proof_calls'][4].pop('host_preparation_status'))

    def test_synthetic_fast_path_explicitly_labeled(self):
        self.assertEqual(self.check(verify_files=False)['status'], 'PASS_SYNTHETIC_CONTRACT_ONLY')


if __name__ == '__main__':
    unittest.main()
