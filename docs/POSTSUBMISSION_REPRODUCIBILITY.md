# Post-submission benchmark: source, prerequisites and execution

This guide describes the launched `PCI_Postsubmission_Certified_v1` source without changing it. The public repository is a source/evidence-index release, not a self-contained archive of all fifteen frozen checkpoints. The original submitted files and completed studies remain separate.

## Read-only inspection first

From the repository root:

```bash
python3 scripts/verify_sources.py
python3 scripts/verify_postsubmission_sources.py
```

The second verifier checks six launched package payloads, four prototype scripts, and five sensitivity-analysis source/input files. It parses source syntax and checks fixed protocol arithmetic without importing the training/prover pipeline. The manifests cover source/input identities, not scientific correctness.

The experiment's `SOURCE.sha256` was checked on the active server and against the released package. Its seven-file source ZIP identity is:

```
389920c5286184e223fe3a76e73763d45c3f1dceee56174e081ace9df76e1b94
```

## Required existing artifacts

The run requires the two original archives identified in [ARTIFACTS.md](ARTIFACTS.md), the fifteen base two-door checkpoints (PCI, fixed penalty and consensus; seeds 0–4), their original configurations/completion records, and the compatible base sources. Restore the base result tree under `results/deadline/`; inspect the manifest before extracting. Never bypass the original archive or checkpoint identity checks.

It also needs a genuine validated blinded variant, the compatible Circom/Node/snarkjs dependencies, the original installed Node lock, and a CPU Python environment with the follow-up's dependencies (RWARE is used by the component profiler). Direct versions are specified in `experiments/PCI_AAMAS27_Followup_v1/requirements.txt`. The existing launch uses PyTorch 2.10.0+cpu. A source-only clone is insufficient for the full benchmark; the missing primary archives are not publicly downloadable from this update.

## Recreate the two-agent functional prerequisite in a new directory

The following shell variables must name existing restored inputs and an unused output path. Do not execute these steps against the active experiment or overwrite a completed variant.

```bash
export REPO="$(pwd)"
export PCI_BASE="/absolute/path/to/restored/PCI_AAMAS27_Rebuild_v1"
export PCI_FOLLOWUP="/absolute/path/to/restored/PCI_AAMAS27_Followup_v1"
export PCI_PYTHON="$PCI_FOLLOWUP/.venv/bin/python"
export PCI_BLINDED_OUT="/absolute/path/to/new/PCI_Blinded_Action_v2"

bash "$REPO/experiments/PCI_Blinded_Action_Prototype_v2/validate_on_server.sh"
```

The script refuses an existing destination, verifies source prerequisites, prepares the private-action-opening variant, builds a new two-agent circuit and keys, and runs host and genuine circuit/binding checks. Its marker is `CRYPTO_SMOKE_COMPLETE_NOT_BENCHMARKED`. This prerequisite is not the full study.

The underlying tools are Circom 2.2.2, circomlib 2.0.5, circomlibjs 0.1.7 and snarkjs 0.7.5; a compatible Node runtime and pinned installation are required. Artifact hashes must correspond to the same compiled circuit. Reusing the public phase-1 transcript is not reusing an old circuit's keys.

## Full workflow in a separate working copy

Copy only the seven-file experiment directory into a new working directory, outside any active run. Point `PCI_OLD_BLINDED` at the successfully validated variant and use the restored inputs above:

```bash
export PCI_OLD_BLINDED="$PCI_BLINDED_OUT"
export WORK="/absolute/path/to/new/PCI_Postsubmission_Certified_v1"
test ! -e "$WORK" || { echo "Destination exists; inspect it first"; exit 2; }
cp -R "$REPO/experiments/PCI_Postsubmission_Certified_v1" "$WORK"
cd "$WORK"
mkdir -p validation
nohup nice -n 10 bash launch.sh > validation/launch.log 2>&1 < /dev/null &
tail -n 40 -F validation/launch.log
```

The preserved launcher targets Linux with `flock`, `taskset`, system-inspection tools, and `nvidia-smi` available. It reads GPU status but performs no CUDA work. Because it uses `set -e`, a missing inspection command also stops the launcher. This is the tested deployment wrapper, not a universal macOS or CPU-only-machine installer. The Python stages can be invoked explicitly in a separately documented replication environment; do not silently patch the frozen wrapper and call that an identical deployment.

The launcher runs prepare, host tests, CPU component profiling, real cryptographic preflight, nine smoke cases, the full 270-case protocol, complete-data analysis, and verified packaging. It never starts the proposed fresh-seed learning study. Preparation copies scientific sources to `variant/`, reuses dependency links by identity, and constructs fresh keys for each population. The active source fingerprint includes its README and copied variant sources; changing documentation inside a working run can invalidate resumption.

## Inspect the existing run without relaunching

On its host, from the already running experiment directory:

```bash
tail -n 40 -F validation/launch.log
"$PCI_PYTHON" benchmark.py status
```

The latter is read-only when successful; its common exception handler records an error state if status parsing fails. Ctrl+C stops the log viewer, not the detached controller. Do not launch a second copy or change a source file to recover from a failed identity guard. Individual `CERTIFIED` and `EPISODE_COMPLETE` lines are not whole-study success.

Require all of the following before reporting full completion: 270 complete cases, complete analysis with the expected counts, unchanged source/checkpoint identities, the final archive, successful member verification, and the archive checksum. The final marker is:

```
POSTSUBMISSION_FULL_BENCHMARK_ARCHIVE_VERIFIED
```

The resulting files are `PCI_postsubmission_certified_results.tar.gz` and `PCI_postsubmission_certified_results.tar.gz.sha256`. Check the copied checksum independently before deleting any server data. No final checksum is claimed before that archive exists.

## Interpretation and privacy boundaries

Every transition uses Groth16 admission. PLONK and the two signature comparators are paired measurements on the same selected records, not separate policies or independently executed episodes. Episode wall time includes all sampled backends; it is not Groth16-only deployment time. The signed-assertion verifier checks a signature/expected statement rather than reevaluating the predicate. Keys must be trusted through an external deployment arrangement; no PKI is implemented.

The private witness, signed disclosure, and SNARK-only streams are intentionally separate. The full synthetic research archive reveals openings and must not be presented as the auditor's restricted view. Random field openings, finite rejection tests, and complete episodes do not establish a general privacy theorem. Hardware load is measured, so do not interpret a difference from old timings as a controlled speedup. Five old training seeds and two matched evaluation seeds per cell do not constitute a new confirmatory learning population.
