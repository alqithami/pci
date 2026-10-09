# Artifact identities and release boundaries

## Included source and data

The two original experiment directories retain their 52 scientific source files and original repository manifests (35 base, 17 follow-up). The new release adds the exact seven-file post-submission package, four exact blinded-action preparation/validation scripts, and the completed nominal sensitivity analysis's scripts and sufficient statistics. Repository-level guides, status snapshots and additional checksum manifests are packaging additions, not edits to an active scientific run.

Existing result CSVs remain unchanged rounded presentation extracts. The new `analysis/nominal_sensitivity_20261009/` inputs are exact event totals for 125 nominal policies and separate certified-intervention counts, not the complete raw archive. They reproduce all 40 nominal endpoint comparisons without training. The integer count JSON is whitespace-compacted but has the same canonical digest as the final supplement:

```
91c68f761cdc483beddf1622b36e70e91fca91314557126af0a25baf77531524
```

## Completed full base archive — not uploaded here

```
Filename: PCI_deadline_results.tar.gz
SHA-256: bc7044ba38edea7fb580c51a1228e56ea403ed18091a09a9e86b374c7f88d381
Completed UTC: 2026-10-08T01:18:20Z
Manifest-listed payloads: 3447
Recorded base source digest: 5b50131d96156bf0d9c35e7ad4af1a074a85bca0185cfc5ddd5125ad0a8b7e01
```

The approximately 415 MB archive contains per-run results, checkpoints, synthetic witnesses, proofs/public vectors, generated circuits, keys and installed dependency locks. It remains author-held; no GitHub release asset or public download URL is claimed. The source-only export does not reproduce its complete source fingerprint until exact installation-generated files are restored. Do not weaken the identity guard to resume a run.

## Completed full learning-follow-up archive — not uploaded here

```
Filename: PCI_followup_results.tar.gz
SHA-256: 45f52edf232c6162fba4d6e3168ef02e4c12cb0abd53c46a4b73b5f8f9998708
Bytes: 532707227
Completed UTC: 2026-10-08T08:46:26Z
Manifest-listed payloads: 18663
Scientific source digest: ba8d57b73079993a77269e357094e4d7c23694b16a392f0ccc1be9dc4ca79b5e
```

The copied archive and all listed payloads were checked. Separate reconstruction reproduced the original statistics, and the saved SNARK proofs and transition bindings were checked. This is internal artifact validation, not independent retraining or a formal security audit. Ed25519 signature bytes were not retained in this original follow-up. See [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md) for corrected inference, metric and privacy scope.

## Source distributions

The original follow-up source ZIP `PCI_AAMAS27_Followup_v1_code.zip` has SHA-256:

```
7f00712c2f7fdab955570a1d1033d711cb350bc58135c6981851c901b31370ff
```

The new seven-file distribution `PCI_Postsubmission_Certified_v1_code.zip` has SHA-256:

```
389920c5286184e223fe3a76e73763d45c3f1dceee56174e081ace9df76e1b94
```

Its payload source checksums are preserved in `experiments/PCI_Postsubmission_Certified_v1/SOURCE.sha256`. All six listed payloads were checked on the running server; the manifest is the seventh file. The protocol's byte identity is:

```
82ac42a23aac5aacf42bbb9cad6b1dac78e79826392e57a698af8e3d0178c4d1
```

The prototype's new repository checksum manifest covers its four unchanged scripts, excluding historical local reports and the new README. It is an export manifest, not the original ZIP's full manifest. The additional source verifier does not replace any experiment's runtime source fingerprint.

## Running post-submission archive — not ready

The target output is `PCI_postsubmission_certified_results.tar.gz`. No checksum, public download, complete data coverage, or successful full-result audit is claimed before packaging finishes. [POSTSUBMISSION_STATUS.json](POSTSUBMISSION_STATUS.json) is a dated progress snapshot only. The final gate is `POSTSUBMISSION_FULL_BENCHMARK_ARCHIVE_VERIFIED`, after complete analysis and member/whole-archive checks.

That future full archive intentionally contains synthetic private witnesses/openings, disclosed records, SNARK-only public views, signatures, keys and trusted checkpoints. The complete research archive is not an auditor-only privacy view. Publication of any final data subset must preserve that distinction and the difference between generation, verification, admission and complete-episode wall times.

The current release contains no new interim outcome tables, raw witnesses, signing secrets, proving keys, live logs, manuscript files, confidential reviews or private connection information. A status count is not a scientific result. No new software license is assigned.

## Restore and verify safely

Check the expected archive checksum and inspect `bundle_manifest.json`; reject unsafe absolute/traversal paths, links and duplicate names. Restore into a new directory, then check every named member, configurations, checkpoints, circuit/key identities and public-signal order. Never overwrite a running study or silently substitute fresh-run outcomes into original evidence. Only load trusted, identity-verified checkpoints; hashes do not make arbitrary pickles safe.

The base archive separates `source/` and `results/`; frozen code expects restored base outcomes under `results/deadline/`. Other archives use their own layouts. Inspect each manifest rather than assuming one layout. Follow [POSTSUBMISSION_REPRODUCIBILITY.md](POSTSUBMISSION_REPRODUCIBILITY.md) for the exact prerequisite chain. Source-verification scripts start no experiments and provide no guarantee of scientific correctness or application-level privacy.
