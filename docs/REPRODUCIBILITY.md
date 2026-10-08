# Reproduction guide

## Three different operations

**Inspect the released source:** clone and run the standard-library manifest verifier. This starts no training and requires no Node or cryptographic toolchain.

**Make a fresh experimental replication:** install the recorded dependencies, validate a new environment, and run the base study in a new results directory. New setup randomness, dependencies, and hardware can change hashes or timings. A fresh run must not be presented as the original archived run.

**Resume or audit frozen evidence:** use the original checkpoint, configuration, installed lockfiles, keys, and source fingerprint. Do not copy new files into a running experiment, change its manifest, bypass a changed-source guard, or merge old and new CSVs.

## Source verification and Python tests

From the repository root:

```bash
python3 scripts/verify_sources.py
export REPO="$PWD"
export PCI_BASE="$REPO/experiments/PCI_AAMAS27_Rebuild_v1"
export FOLLOWUP="$REPO/experiments/PCI_AAMAS27_Followup_v1"
```

For an isolated Python 3.12 CPU environment:

```bash
python3.12 -m venv "$REPO/.venv"
export PY="$REPO/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install -r "$FOLLOWUP/requirements.txt"
PYTHON="$PY" bash scripts/run_python_tests.sh
```

The test wrapper invokes the base and follow-up suites separately, without skipping missing RWARE dependencies. These are 57 base and 29 follow-up Python test cases in the supplied snapshots. Import failure is an error, not a pass. Python tests include small optimizer checks but do not replace real Circom/snarkjs verification.

No GPU, Watsonx account, or proprietary data is required by these CPU implementations. These instructions do not install onto or modify an existing remote experiment.

## Cryptographic toolchain

The recorded scientific versions are Circom 2.2.2, circomlib 2.0.5, circomlibjs 0.1.7, and snarkjs 0.7.5. Install Node and the compiler before running crypto tests. `crypto/package.json` pins direct Node dependencies. Exact installation-generated transitive locks are retained in the original full archive, not recreated in this source release.

The archived base `setup_ubuntu.sh` is an Ubuntu installer that uses sudo/apt, installs Python/Node tooling, and builds Circom with Rust. Read it before running it. It is not a transcript of the already provisioned experiment server, whose toolchain was installed separately. Its full clean-machine path has not been re-executed as part of this repository update.

With a compatible compiler available on PATH (or `CIRCOM` pointing to it) and Node installed:

```bash
cd "$PCI_BASE/crypto"
# For exact archived dependencies, restore the original package-lock.json and use npm ci.
# For a fresh replication without that lock, npm install creates a new recorded lock.
npm install
cd "$PCI_BASE"
"$PY" -m pci_bench.crypto_build --agents 2 4 8
"$PY" -m pci_bench.crypto_check --out results/new_preflight/crypto
```

The builder downloads and verifies a public Powers-of-Tau transcript and generates keys. This consumes time/storage and creates new setup randomness. Never replace a frozen key or reuse a key from a different circuit. The generator, circuit source, public-signal ordering, and artifact hashes must agree.

## Fresh base study

Only run the expensive profile after the real smoke test succeeds:

```bash
cd "$PCI_BASE"
PCI_PYTHON="$PY" bash run_all.sh --profile smoke --jobs 2 --results results/smoke
PCI_PYTHON="$PY" bash run_all.sh --profile deadline --jobs 12 --results results/deadline
"$PY" scripts/status.py results/deadline
"$PY" scripts/package_results.py results/deadline
sha256sum -c PCI_deadline_results.tar.gz.sha256
```

`--core-only` is a diagnostic smoke option, not permission to claim cryptographic completion. `COMPLETE` requires the requested suite; `CORE_SMOKE_ONLY` and partial analysis do not establish full evidence.

Choose worker counts appropriate to the available CPU/memory and other workloads. No runtime estimate is a scientific guarantee. A complete run must include all fixed conditions, not only successful or favorable ones.

## Follow-up prerequisites and portable execution

The follow-up imports the intact base `pci_bench` through `PCI_BASE`. Its certificate phase needs the base's two-door PCI checkpoints at seeds 0 and 1, plus matching generated circuits, verification/proving keys, and installation lockfiles. A source-only clone is insufficient to reproduce the original certificate measurements. Either restore the verified original archive or finish a separately identified fresh base replication.

For exact original evidence, preserve the source digest recorded in its completion marker; the installed Node lock contributes to that digest. The new repository's subset manifests do not replace that run fingerprint. See [ARTIFACTS.md](ARTIFACTS.md).

The following commands allow a new working location without editing the frozen Python files:

```bash
export PCI_BASE="$REPO/experiments/PCI_AAMAS27_Rebuild_v1"
cd "$FOLLOWUP"
"$PY" -u -m pci_followup.suite --smoke --jobs 2 --results results/smoke
"$PY" -u -m pci_followup.suite --jobs 12 --results results/full
"$PY" -m pci_followup.status
"$PY" -m pci_followup.package
sha256sum -c PCI_followup_results.tar.gz.sha256
```

The preserved `launch_server.sh` is the original deployment-specific controller: it assumes `/home/ubuntu`, an installed `uv`, project-local environments, and the exact original archive. It is **not** the portable entry point for a source-only clone. The commands above use `PCI_BASE` and the selected Python explicitly. Do not run either workflow against the currently active experiment directory.

## Results and security

Both studies write strict per-run schemas, immutable configurations, checkpoints, separate evaluation outputs, and completeness records. The follow-up records normalized variants separately from the base. Its package includes synthetic private witnesses for research reproduction; do not apply those archive defaults to confidential inputs.

PyTorch checkpoints used here contain optimizer/RNG/environment objects and are loaded with `weights_only=False` where required. Only load trusted, verified experiment checkpoints. A checksum establishes identity relative to the expected file; it does not turn an untrusted pickle into safe code.

Source tests, cryptographic preflight, completed training, complete evaluation, successful analysis, and verified packaging are distinct milestones. The paper's substantive claims require inspecting the complete outcomes, not only these success markers.
