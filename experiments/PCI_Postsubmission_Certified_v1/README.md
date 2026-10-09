# PCI post-submission repaired-circuit benchmark

This is a **new post-submission study**, not a replacement for either completed
learning study or the submitted manuscript. It evaluates the domain-1004 private
**action-opening** repair over complete episodes and directly compares frozen
PCI, fixed-penalty, and full-action-consensus policies. No policy is retrained.

## Fixed matrix

Three policy families x five archived training seeds x three populations (2/4/8)
x three rule regimes (nominal/tight quota/rule shift) x two matched new evaluation
episodes = **270 full 256-step episodes**. There are **69,120 Groth16-gated
transitions**. At 24 fixed indices per episode, all four backends are measured in
balanced order: Groth16, PLONK, signed assertion, and authenticated disclosure.
This gives **6,480 paired records, 88,560 backend measurements, and 75,600 SNARK
proofs**. Full traces and signature bytes/public keys are retained for replay.
Private openings are sampled from the operating system, independently of the
policy and task random streams. Fixed small openings are not used in the benchmark.

The machine-readable protocol freezes the matrix, endpoints, analysis, seeds,
resource limits and assumptions. All negative, null, failed, and zero-delivery
outcomes are retained. Interrupted cases are preserved in separate directories;
repeated attempts are not silently merged. There is no best-checkpoint selection.

## Execution and gates

The launcher checks original archive and checkpoint hashes, copies the already
validated blinded scientific sources into a new directory, and generates **new
keys** for every tested population. It does not modify old source or keys. Shared
Node/compiler/public-phase-1 resources are reused by identity, not old proving
keys. Host tests include 2,304 actual simulator transitions against a separate
scalar oracle. Real preflight uses both SNARKs on honest, unsafe, substituted,
range-alias and replay cases. A nine-episode, four-step integration smoke covers
all three methods and populations before allowing full experiments.

The full controller re-verifies every persisted proof and signature after each
episode and checks state/count/reset continuity. It performs complete-data-only
analysis, then archives sources, checkpoints, keys, witnesses, public views and
results. Every member and the archive checksum must pass before the final marker.
`SMOKE_COMPLETE` is not the full completion marker.

The primary operational family contains 36 comparisons: PCI versus penalty and
consensus x three populations x three regimes x throughput and institutional
replacement rate. The two evaluation episodes average within each archived
training seed before inference. Paired t intervals, descriptive bootstrap and
sign-flip sensitivity remain assumption-dependent with only five trained seeds;
Holm correction is applied separately to the two test families. This is NOT a new
20-30-seed confirmatory learning study.

## Timing and trust boundaries

One sequential prover worker is limited to six CPU cores at lower scheduling
priority. Existing PBRC/GPU work is not stopped or preempted. Host load is recorded;
these are shared-machine timings, not a controlled hardware speedup comparison to
previous runs. Generation, verification, per-call wall time, record binding and
complete-episode time remain distinct. The whole episode includes paired extra
backends, so it is not Groth16-only runtime. Record binding is repeated in each
paired measurement for traceability; do not sum it four times.

The signed-assertion comparator checks an authenticated claim, not the predicate;
it needs a trustworthy signer for predicate assurance. The disclosed-record
comparator reveals record and openings, authenticates them, reconstructs expected
commitments, and reevaluates the predicate. The SNARK checks the encoded relation
against authentic expected commitments. A finite false-assertion fixture shows
that a valid signature can authenticate an incorrect claim, not that Ed25519 is
broken. State authority, authentic expected commitments and faithful executor
remain trusted; no sensor attestation, malicious multiparty proving, inference
proof, or crash-persistent transactional ledger is implemented.

SNARK-only public views exclude private openings. Disclosed and private archive
views intentionally expose them for this synthetic reproduction. Combining these
views would invalidate any claim about SNARK-only disclosure. Functional tests
and randomized openings do not prove general trajectory confidentiality; auxiliary
information, timing and metadata require a separate privacy argument. Rules remain
public. No empirical action-recovery attack rate is claimed by this package.

The separate CPU profile uses real simulator steps and forward/backward
microbenchmarks, not new learning outcomes. GPU profiling and a fresh confirmatory
learning protocol are subsequent tasks; this launcher does not silently start a
large GPU grid or compete with the existing GPU application.

## On the existing IBM server

This directory is intended for `/home/ubuntu/PCI_Postsubmission_Certified_v1`.
The default Python is the existing PCI follow-up CPU environment. It is used
read-only; no dependency installation or CUDA allocation is performed.

```bash
cd /home/ubuntu/PCI_Postsubmission_Certified_v1
nohup nice -n 10 bash launch.sh > validation/launch.log 2>&1 < /dev/null &
# Read-only status:
/home/ubuntu/PCI_AAMAS27_Followup_v1/.venv/bin/python benchmark.py status
# Live progress (Ctrl+C stops viewing, not the run):
tail -n 40 -F validation/launch.log
```

Duplicate launches are blocked by a controller lock. Failed prerequisites stop
completion. Only the following indicates the complete study and verified archive:

```text
POSTSUBMISSION_FULL_BENCHMARK_ARCHIVE_VERIFIED
```

The final files are `PCI_postsubmission_certified_results.tar.gz` and its `.sha256`.
Do not remove server files until the downloaded archive is independently verified.
Full archives contain trusted Python checkpoints loaded with `weights_only=False`;
never substitute unknown checkpoint files. No results are yet claimed here.
