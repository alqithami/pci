# Artifact identity, availability, and source preservation

## Included in this repository

Two versioned scientific source directories, execution protocols, tests, circuit generation/verification and analysis/packaging tools. Separate base and follow-up directories contain audited **presentation extracts**, not full raw results. Follow-up audit counts and interpretation are now included.

The source export manifests cover 35 base files and 17 follow-up files. They exclude manuscript drafts, private deployment notes, installed dependency locks and heavy artifacts. Each listed scientific source preserves its supplied source-package bytes. This results/documentation update does not modify those files or their manifests.

The follow-up source digest, including its protocol and the 15 imported base Python modules, is:

```
ba8d57b73079993a77269e357094e4d7c23694b16a392f0ccc1be9dc4ca79b5e
```

The standard-library verifier reconstructs that fingerprint; it is not a Git commit ID or the complete base-run fingerprint.

## Completed full base archive — author-held, not uploaded here

```
Filename: PCI_deadline_results.tar.gz
SHA-256: bc7044ba38edea7fb580c51a1228e56ea403ed18091a09a9e86b374c7f88d381
Completed UTC: 2026-10-08T01:18:20Z
Manifest-listed payloads: 3447
Recorded base source digest: 5b50131d96156bf0d9c35e7ad4af1a074a85bca0185cfc5ddd5125ad0a8b7e01
```

The approximately 415 MB archive holds raw per-run records, checkpoints, synthetic witnesses, proofs/public vectors, generated circuits, keys, installed lockfiles and the bundle manifest. No GitHub release asset or public download URL is claimed here. A source-only export does not reproduce its complete fingerprint until exact archived installation-generated files are restored. Never weaken a changed-source check to resume a frozen run.

## Completed full follow-up archive — author-held, not uploaded here

```
Filename: PCI_followup_results.tar.gz
SHA-256: 45f52edf232c6162fba4d6e3168ef02e4c12cb0abd53c46a4b73b5f8f9998708
Bytes: 532707227
Completed UTC: 2026-10-08T08:46:26Z
Manifest-listed payloads: 18663
Scientific source digest: ba8d57b73079993a77269e357094e4d7c23694b16a392f0ccc1be9dc4ca79b5e
```

The server archive and copied archive passed independent checksums, and every listed payload was rehashed. Separate post-run reconstruction checked all evaluations and original bootstrap intervals. All 3,216 saved SNARK proofs were reverified with unchanged keys and independently recomputed bindings; scalar transition checks cover the complete 3,072-step certified sample. See [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md) for scope and limitations.

Full-precision audit outputs and helpers remain alongside the author-held evidence. The public CSVs are rounded extracts. The complete archive has not been added as a GitHub release asset in this update. Ed25519 signature bytes were not retained for post-run replay; do not claim independent replay of those signatures.

## Source distribution identity

The exported source comes from `PCI_AAMAS27_Rebuild_v1.zip` and `PCI_AAMAS27_Followup_v1_code.zip`. The latter distribution has SHA-256:

```
7f00712c2f7fdab955570a1d1033d711cb350bc58135c6981851c901b31370ff
```

Original package manifests include additional deployment/private-document files excluded from Git. Repository manifests explicitly identify the smaller export instead of pretending those absent files are included.

## Verification and restoration

Verify archive SHA-256, inspect `bundle_manifest.json`, and hash every named member before use. Reject absolute/traversal paths, symlinks and duplicates. Restore into a separate directory and never overwrite a running study. Verify configurations, checkpoints, circuit/key hashes, public-signal ordering and completion markers. Hash identity is not a security guarantee for arbitrary untrusted pickled checkpoints.

The base archive separates `source/` and `results/`, with its frozen certification code expecting restored base results under `results/deadline/`. The follow-up archive retains its own paths, frozen base code/keys, and selected frozen checkpoints; inspect its manifest rather than assuming the base archive layout applies unchanged.

The source packaging scripts demonstrate member-by-member verification, not a generic safe extractor for arbitrary untrusted archives. Fresh replications are separate studies; outcomes and keys should not be silently substituted into the frozen original evidence.
