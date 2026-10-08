# Artifact identity, availability, and source preservation

## Included in this repository update

Two versioned scientific source directories, their execution protocols, Python tests, circuit generation/verification code, and analysis/packaging tools. The base-study presentation data include all nominal comparators, paired effects, the matched coefficient sweep, and descriptive certificate medians.

The source manifests are **repository-export manifests**: they cover 35 base files and 17 follow-up files. They exclude original manuscripts, private operational instructions, historical validation reports, installation-generated dependency locks, and heavy artifacts. Each listed scientific file preserves the corresponding supplied source-package bytes. Added repository documentation and these export manifests are not changes to the running experiment.

The follow-up's scientific source digest, including its protocol and the 15 imported base Python modules, is:

```text
ba8d57b73079993a77269e357094e4d7c23694b16a392f0ccc1be9dc4ca79b5e
```

The standard-library verifier reconstructs this 32-input digest as an additional check. It is distinct from Git commit IDs and from the complete base-run fingerprint.

## Completed full base archive — not uploaded by this update

```text
Filename: PCI_deadline_results.tar.gz
SHA-256: bc7044ba38edea7fb580c51a1228e56ea403ed18091a09a9e86b374c7f88d381
Completed UTC: 2026-10-08T01:18:20Z
Manifest-listed files: 3447
Recorded base source digest: 5b50131d96156bf0d9c35e7ad4af1a074a85bca0185cfc5ddd5125ad0a8b7e01
```

That approximately 415 MB archive contains raw per-run outputs, checkpoints, synthetic witnesses, proof/public vectors, generated circuits, keys, installed lockfiles, and its bundle manifest. It remains author-held; no GitHub release asset or public download URL is claimed in this repository update. The summary CSVs are not a substitute for it. Full raw-artifact distribution remains a separate release step.

The supplied source distribution does not contain the installed Node lockfile. Consequently, the source-only export is not asserted to reproduce the complete base-run source digest until the exact archived installation-generated files are restored. Do not weaken that digest check to resume a run.

## Supplied source distribution identity

The exported code is sourced from `PCI_AAMAS27_Rebuild_v1.zip` and `PCI_AAMAS27_Followup_v1_code.zip`. The follow-up distribution SHA-256 is:

```text
7f00712c2f7fdab955570a1d1033d711cb350bc58135c6981851c901b31370ff
```

The original archive-level manifests contain additional deployment/private-document files excluded from Git. The export manifests deliberately identify their smaller scope instead of retaining a manifest whose files are absent.

## Verification once the raw archive is available

First compare the archive SHA-256 with the identity above. Then inspect its `bundle_manifest.json` and hash every named member before using it. Check exact configuration, checkpoint and circuit/key hashes, public-signal ordering, and completeness records. Archive layout separates `source/` and `results/`; frozen certificate code expects the restored base run under `results/deadline/`.

Use a separate restoration directory and reject unsafe absolute, traversal, or symlink archive members. Do not overwrite a running experiment. The preserved packaging scripts demonstrate member-by-member hash verification, but are not generic safe extractors for arbitrary untrusted archives.

## Follow-up evidence

The follow-up was active at the release's last operational check. It is not marked scientifically complete here and its provisional outputs are not published. Completed follow-up results require all fixed runs/evaluations, the longer certificate study, full analysis, archive verification, and a subsequent audit. Its planned counts must never be inserted as observed results.
