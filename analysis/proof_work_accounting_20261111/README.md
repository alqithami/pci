# Independent proof-work ledger counter

`check_proof_ledger.py` checks identities and count partitions against retained
artifact bytes and a separately executed actual-vkey/native audit. It never
executes or simulates a proof verifier. Trusted audit provenance and honest
instrumentation are explicit premises; hashes are not hardware attestation.

The normalized ledger schema is `PCI-PROOF-WORK-LEDGER-v1`. Its lists are
`offers`, `proof_calls`, `preproof_rejections` and `applications`, with an
explicit Boolean `complete_drain`. Proof calls carry unique call/attempt IDs,
their offered ID, and status `VALID`, `FAILED_NO_VALID` or `INFLIGHT`.
Each returned body has a relative `return_artifact` path/byte/hash descriptor;
prepared bodies additionally have `prepared_artifact`. A valid returned proof
whose host encoding fails before preparation remains valid and unapplied; its
status is `VALID`, `host_preparation_status` is `FAILED_AFTER_VALID_PROOF`, and
its prepared artifact is absent. These generated proofs must not disappear.

The audit schema is `PCI-INDEPENDENT-PROOF-CORPUS-AUDIT-v1`, with status
`PASS_ACTUAL_VKEY_NATIVE_REPLAY`, the verification-key hash and
`source_identity.auditor_source_sha256`. Each audit record binds exact call and
attempt IDs, return-artifact hash, validity, native-binding validity, and the
prepared-artifact hash (null if preparation failed). Key and auditor source
hashes are explicit caller trust pins. Every return is covered exactly once.

Fresh applications increment the observed executor prefix by one. Exact
duplicate retries keep that prefix unchanged and create no new effect or
commit. An old attempt can be retried after later commits; its unchanged current
prefix may exceed its original acknowledgment sequence. If supplied, the retry
`ack_sequence` must still match the original application. Fresh task units are
counted once, from distinct applied attempts.

Derived partitions are:

- invoked proof RPCs = valid generated + failed before valid + inflight;
- valid generated = distinct durably applied proof attempts + valid unapplied.

Prefilter rejection is disjoint from proving, duplicates do not create new
proofs or effects, and valid-stale counts require the same audited retained
proof identity. A closed drain refuses an inflight call. This module does not
decide scientific cohort admission or turn a failed run into a complete one.

Run the synthetic contract and real-file byte-binding controls:

```sh
python -B run_qualification.py
```

They execute 37 controls and explicitly claim no genuine cryptographic result.
The fixture's five proofs are synthetic records, not generated proofs. Run the
checker on an actual separately audited corpus with:

```sh
python -B check_proof_ledger.py --ledger ledger.json --proof-audit audit.json \
  --corpus-root actual_corpus --vkey-sha256 ACTUAL_KEY_SHA \
  --auditor-sha256 ACTUAL_AUDITOR_SOURCE_SHA --out fresh_counts.json
```

The Python API is `check_ledger(ledger, proof_audit, corpus_root=...,\
expected_vkey_sha256=..., expected_auditor_sha256=...)`. The explicit
`verify_files=False` path is for synthetic contract qualification only and
returns a synthetic status. All returned receipts set `scientific_result=false`.

## Public artifact scope

This directory publishes the generic counter and its 37 synthetic contract and
retained-byte controls. It contains no genuine proof corpus, private witness,
private packet, operational ledger, consumer database, verification key, or
auditor provenance. The synthetic records are explicitly not generated proofs.

A separately produced real audit must be supplied under independently trusted
verification-key and auditor-source pins before using the counter on genuine
proof outcomes. A matching hash checks retained-byte identity; it does not
establish that the audit was executed honestly or that application records are
complete. Those are explicit instrumentation and consumer-history premises.

This is an engineering measurement contract. It reports neither policy-dependent
proof savings nor scientific learning results. The original training sources,
inference recipes and statistical families are unchanged.
