"""Strict identity and partition checks for a retained genuine-proof work ledger.

This module does not execute a verifier. It binds each claimed proof outcome to
the independently executed audit and exact retained bytes. Audit provenance and
honest instrumentation remain explicit premises, not remote attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


SHA = re.compile(r"[0-9a-f]{64}\Z")


class LedgerError(ValueError):
    pass


def need(value, message):
    if not value:
        raise LedgerError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def sha_field(value, label):
    need(isinstance(value, str) and SHA.fullmatch(value), 'INVALID_SHA:' + label)
    return value


def identifier(value, label):
    need(isinstance(value, str) and value and len(value) <= 256, 'INVALID_ID:' + label)
    return value


def artifact(item, root, *, verify_files):
    need(isinstance(item, dict), 'ARTIFACT_DESCRIPTOR')
    name = item.get('path')
    need(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
         'UNSAFE_ARTIFACT_PATH')
    rel = Path(name)
    need(rel.parts and not rel.is_absolute() and '..' not in rel.parts, 'UNSAFE_ARTIFACT_PATH')
    need(type(item.get('bytes')) is int and item['bytes'] > 0, 'ARTIFACT_SIZE')
    sha_field(item.get('sha256'), 'artifact')
    if not verify_files:
        return
    need(root is not None, 'CORPUS_ROOT_REQUIRED')
    need(not Path(root).is_symlink(), 'SYMLINK_CORPUS_ROOT')
    root = Path(root).resolve()
    path = root / rel
    for p in [path, *path.parents]:
        if p == root.parent:
            break
        need(not p.is_symlink(), 'SYMLINK_ARTIFACT_PATH')
        if p == root:
            break
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        named_before = path.lstat()
        fingerprint = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink,
            s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink > 0 and
             fingerprint(before) == fingerprint(named_before), 'ARTIFACT_HELD_IDENTITY')
        h = hashlib.sha256(); size = 0
        for body in iter(lambda: handle.read(1 << 20), b''):
            h.update(body); size += len(body)
        after = os.fstat(handle.fileno()); named_after = path.lstat()
        need(fingerprint(before) == fingerprint(after) == fingerprint(named_after),
             'ARTIFACT_CHANGED_DURING_READ')
    need(size == item['bytes'] and h.hexdigest() == item['sha256'], 'ARTIFACT_HASH_OR_SIZE')


def check_ledger(ledger, proof_audit, *, corpus_root=None, verify_files=True,
                 expected_vkey_sha256, expected_auditor_sha256):
    """Validate the normalized v1 contract and return only derived counts.

    ``verify_files=False`` is restricted to synthetic contract qualification;
    that result always has ``scientific_result=False`` and a synthetic status.
    """
    need(ledger.get('schema') == 'PCI-PROOF-WORK-LEDGER-v1', 'LEDGER_SCHEMA')
    need(proof_audit.get('schema') == 'PCI-INDEPENDENT-PROOF-CORPUS-AUDIT-v1', 'AUDIT_SCHEMA')
    need(proof_audit.get('status') == 'PASS_ACTUAL_VKEY_NATIVE_REPLAY', 'AUDIT_NOT_PASS')
    vkey = sha_field(expected_vkey_sha256, 'expected_vkey')
    auditor = sha_field(expected_auditor_sha256, 'expected_auditor')
    need(proof_audit.get('vkey_sha256') == vkey, 'AUDIT_VKEY_IDENTITY')
    need(isinstance(proof_audit.get('source_identity'), dict) and
         proof_audit['source_identity'].get('auditor_source_sha256') == auditor, 'AUDIT_SOURCE_IDENTITY')
    for key in ('offers', 'proof_calls', 'preproof_rejections', 'applications'):
        need(isinstance(ledger.get(key), list), 'LEDGER_LIST:' + key)
    need(type(ledger.get('complete_drain')) is bool, 'DRAIN_DECLARATION')
    offers = {}
    for row in ledger['offers']:
        key = identifier(row.get('offered_id'), 'offer')
        need(key not in offers, 'DUPLICATE_OFFER_ID')
        offers[key] = row
    calls = {}; attempts = {}; returned_paths = set()
    for row in ledger['proof_calls']:
        key = identifier(row.get('call_id'), 'call')
        attempt = identifier(row.get('attempt_id'), 'attempt')
        offer = identifier(row.get('offered_id'), 'call_offer')
        need(key not in calls and attempt not in attempts, 'DUPLICATE_PROOF_CALL_OR_ATTEMPT')
        need(offer in offers, 'UNKNOWN_CALL_OFFER')
        need(row.get('status') in ('VALID', 'FAILED_NO_VALID', 'INFLIGHT'), 'PROOF_STATUS')
        if 'return_artifact' in row:
            item = row['return_artifact']; artifact(item, corpus_root, verify_files=verify_files)
            need(item['path'] not in returned_paths, 'REUSED_RETURN_ARTIFACT_PATH')
            returned_paths.add(item['path'])
        if row['status'] == 'VALID':
            need('return_artifact' in row, 'VALID_PROOF_BODY_MISSING')
        if 'prepared_artifact' in row:
            need(row['status'] == 'VALID', 'NONVALID_PROOF_HAS_PREPARED_BODY')
            artifact(row['prepared_artifact'], corpus_root, verify_files=verify_files)
        elif row['status'] == 'VALID':
            need(row.get('host_preparation_status') == 'FAILED_AFTER_VALID_PROOF',
                 'UNPREPARED_VALID_PROOF_STATUS')
        if row['status'] == 'INFLIGHT':
            need('return_artifact' not in row, 'INFLIGHT_HAS_RETURN_BODY')
        calls[key] = row; attempts[attempt] = row
    audit_rows = proof_audit.get('records')
    need(isinstance(audit_rows, list), 'AUDIT_RECORDS')
    audited = {}
    for row in audit_rows:
        key = identifier(row.get('call_id'), 'audit_call')
        need(key not in audited and key in calls, 'DUPLICATE_OR_UNKNOWN_AUDIT_CALL')
        call = calls[key]
        need('return_artifact' in call, 'AUDIT_CALL_HAS_NO_RETURN')
        need(row.get('attempt_id') == call['attempt_id'], 'AUDIT_ATTEMPT_IDENTITY')
        need(row.get('return_artifact_sha256') == call['return_artifact']['sha256'], 'AUDIT_RETURN_IDENTITY')
        need(type(row.get('valid')) is bool, 'AUDIT_VALIDITY_TYPE')
        need(type(row.get('native_binding_valid')) is bool, 'AUDIT_NATIVE_BINDING_TYPE')
        expected_prepared = call.get('prepared_artifact', {}).get('sha256')
        need(row.get('prepared_artifact_sha256') == expected_prepared, 'AUDIT_PREPARED_IDENTITY')
        if call['status'] == 'VALID':
            need(row['valid'], 'CLAIMED_VALID_PROOF_FAILED_AUDIT')
            need(row['native_binding_valid'], 'AUDIT_NATIVE_STATEMENT_BINDING')
        else:
            need(call['status'] == 'FAILED_NO_VALID' and not row['valid'],
                 'RETURNED_VALID_PROOF_MISCLASSIFIED')
        audited[key] = row
    returned = {k for k, row in calls.items() if 'return_artifact' in row}
    need(set(audited) == returned, 'UNAUDITED_RETURN_OR_EXTRA_AUDIT')
    rejected = set()
    for row in ledger['preproof_rejections']:
        offer = identifier(row.get('offered_id'), 'reject_offer')
        need(offer in offers and offer not in rejected, 'UNKNOWN_OR_DUPLICATE_PREPROOF_REJECT')
        need(row.get('proof_called') is False, 'PREFILTER_REJECTION_CLAIMS_PROOF')
        need(not any(call['offered_id'] == offer for call in calls.values()),
             'PREFILTER_REJECT_COUNTED_AS_PROOF')
        rejected.add(offer)
    applied = {}; sequences = {}; duplicates = 0; stale = set(); application_events = []
    for row in ledger['applications']:
        aid = identifier(row.get('attempt_id'), 'application_attempt')
        need(aid in attempts, 'UNKNOWN_APPLICATION_ATTEMPT')
        call = attempts[aid]
        need(call['status'] == 'VALID', 'NONVALID_PROOF_APPLICATION')
        need(row.get('offered_id') == call['offered_id'], 'APPLICATION_OFFER_IDENTITY')
        status = row.get('status')
        need(status in ('APPLIED', 'DUPLICATE', 'REJECTED'), 'APPLICATION_STATUS')
        if status == 'APPLIED':
            need(aid not in applied, 'ATTEMPT_APPLIED_TWICE')
            need('prepared_artifact' in call, 'APPLICATION_PREPARED_BODY_MISSING')
            need(type(row.get('seq_before')) is int and type(row.get('seq_after')) is int
                 and row['seq_before'] >= 0 and row['seq_after'] == row['seq_before'] + 1,
                 'APPLICATION_SEQUENCE_INCREMENT')
            need(type(row.get('domain')) is int and row['domain'] >= 0, 'APPLICATION_DOMAIN')
            domain = row['domain']
            key = (domain, row['seq_after'])
            need(key not in sequences, 'TWO_ATTEMPTS_ONE_DURABLE_SEQUENCE')
            need(type(row.get('task_units')) is int and row['task_units'] >= 0, 'TASK_UNITS')
            applied[aid] = row; sequences[key] = aid
        elif status == 'DUPLICATE':
            need(aid in applied, 'RETRY_WITHOUT_PREVIOUS_APPLICATION')
            original = applied[aid]
            need(type(row.get('seq_before')) is int and type(row.get('seq_after')) is int
                 and row['seq_before'] == row['seq_after'] and row['seq_after'] >= original['seq_after']
                 and row.get('domain') == original.get('domain'),
                 'DUPLICATE_CHANGED_COMMIT_PREFIX')
            need(row.get('new_commit') is False and row.get('new_effect') is False,
                 'DUPLICATE_ADDS_COMMIT_OR_EFFECT')
            if 'ack_sequence' in row:
                need(row['ack_sequence'] == original['seq_after'], 'DUPLICATE_ACK_IDENTITY')
            duplicates += 1
        else:
            need(isinstance(row.get('code'), str) and row['code'], 'REJECTION_CODE')
            if row['code'] == 'STALE_PREDECESSOR_OR_VERSION':
                need(aid not in applied, 'APPLIED_ATTEMPT_CLAIMED_STALE')
                stale.add(aid)
        application_events.append((aid, status))
    valid = {k for k, c in attempts.items() if c['status'] == 'VALID'}
    failed = {k for k, c in attempts.items() if c['status'] == 'FAILED_NO_VALID'}
    pending = {k for k, c in attempts.items() if c['status'] == 'INFLIGHT'}
    need(not ledger['complete_drain'] or not pending, 'COMPLETE_DRAIN_HAS_INFLIGHT_PROOFS')
    need(set(applied) <= valid and stale <= valid - set(applied), 'VALID_PROOF_PARTITION')
    unused = valid - set(applied)
    need(len(calls) == len(valid) + len(failed) + len(pending), 'INVOCATION_PARTITION')
    need(len(valid) == len(applied) + len(unused), 'VALID_GENERATION_PARTITION')
    return {'schema': 'PCI-PROOF-WORK-LEDGER-CHECK-v1',
        'status': 'PASS_RETAINED_ARTIFACT_AND_AUDIT_BINDINGS' if verify_files else 'PASS_SYNTHETIC_CONTRACT_ONLY',
        'scientific_result': False, 'proof_verifier_executed_by_this_module': False,
        'offered': len(offers), 'proof_rpc_invocations': len(calls),
        'valid_generated_proofs': len(valid), 'failed_before_valid': len(failed),
        'inflight': len(pending), 'distinct_durably_applied_proof_attempts': len(applied),
        'valid_unapplied_proofs': len(unused), 'valid_stale_unapplied_proofs': len(stale),
        'preproof_rejections': len(rejected), 'duplicate_retries': duplicates,
        'task_units': sum(r['task_units'] for r in applied.values()),
        'valid_unapplied_attempt_ids': sorted(unused), 'ledger_sha256': digest(ledger),
        'proof_audit_sha256': digest(proof_audit), 'vkey_sha256': vkey,
        'auditor_source_sha256': auditor,
        'scope': 'Counts bind declared independently executed actual-vkey/native audits to retained artifact bytes. This is not a second cryptographic verification or a producer/hardware attestation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--proof-audit', type=Path, required=True)
    p.add_argument('--corpus-root', type=Path, required=True)
    p.add_argument('--vkey-sha256', required=True)
    p.add_argument('--auditor-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    need(not a.out.exists(), 'FRESH_OUTPUT_REQUIRED')
    result = check_ledger(json.loads(a.ledger.read_text()), json.loads(a.proof_audit.read_text()),
        corpus_root=a.corpus_root, expected_vkey_sha256=a.vkey_sha256,
        expected_auditor_sha256=a.auditor_sha256)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('status','proof_rpc_invocations', 'valid_generated_proofs',
          'distinct_durably_applied_proof_attempts','valid_unapplied_proofs')}))


if __name__ == '__main__':
    main()
