# PCI follow-up v1 — post-audit, outcome-independent follow-up protocol

The original 110-run study remains frozen, including its null comparisons. This
study was designed after those outcomes were inspected. It is not a retroactive
preregistration and must never replace them silently.

## Learning matrix

125 new runs: five methods × five configurations × five independent training
seeds (101–105), 262,144 joint steps each, totalling 32,768,000 joint steps.
Configurations: the original three custom layouts and native RWARE 2.0.0 tiny and
small layouts. RWARE uses four agents, eight requested shelves, native five-action
orientation/loading dynamics and native individual delivery reward. It is an
independently implemented warehouse, not a new real-world domain.

All methods receive the same added announced task/map features, context masks,
actor architecture, two critics, optimizer budget, reward shaping, institution
rules, seeds and evaluation. RWARE original observations are not misrepresented
as the enriched inputs used here. Its native environment step/reset are not
patched. Enrichment chooses the nearest requested shelf, native goal, or empty
return location; these cues and dense potential/pickup/return shaping are shared
by every method. Throughput counts only native delivery events (reward=1), never
shaping. The external institutional extension limits simultaneous region entries
and per-agent entry counts; this differs explicitly from the custom world's
standing region capacity. RWARE has no SNARK certificate in this study.

Methods: fixed penalty, trace-normalized full-action consensus, trace-normalized
PCI, spectrum-matched nonsemantic rotation, and an empirical-rate PPO-Lagrangian
adaptation. The adaptive baseline has separate reward/cost critics and GAEs,
pessimistically clipped reward and cost surrogates, and projected multiplier
updates. Every method has both critics, preserving actor and total parameter
counts. The dual update is λ'=clip(λ+0.5*(empirical_cost_rate−0.02),0,10), starting
at 0.25. Costs are mean per-agent attempted wall/conflict/quota indicators plus
allocated aggregate excess, under the corresponding domain's rules. This is an
approximate empirical-rate constrained PPO implementation; no per-iteration
safety or MACPO reproduction claim is made. Discounted GAEs are .99/.95; the dual
update uses the undiscounted rollout-average rate. This estimator distinction is
explicit, not a constrained-convergence theorem.

The regularizer coefficient is 0.2. Center each five-action effect matrix F over
actions and divide squared projected residuals by ||F||_F² (zero map -> zero).
Consensus uses (I−11ᵀ/5), whose squared Frobenius norm is four. A fixed seeded
orthogonal transform of the centered maps preserves each map's singular values,
rank and norm while removing its original action semantics. This matches operator
scale, NOT the evolving policy-gradient norms. Those norms and their ratio to the
actor objective gradient are measured at the first minibatch of every update.
No coefficient or checkpoint is selected using test outcomes.

Every final policy gets 24 matched held-out episodes per nominal/tight-quota/
rule-shift/2-agent/8-agent/12-agent scenario, both admission modes, on evaluation
seeds 410000–410023. Initial and quarter-interval checkpoints get eight separate
nominal unshielded episodes (510000–510007). Primary endpoints are nominal
unshielded actual throughput and proposed joint compliance; the cost budget and
admission intervention outcomes are separate diagnostics. All comparisons and
layouts, including null/failure outcomes, must be retained. Five seeds, not the
number of timesteps or bootstrap draws, determine replication. Intervals are
pointwise exploratory 95% paired training-seed bootstrap intervals, 5000 draws,
seed 92170; no multiplicity-adjusted significance claim.

## Certified execution

Reuse original frozen PCI checkpoints seed 0 and 1, never retrain or select them
based on the new episodes. Populations 2,4,8 × two checkpoints × nominal/rule-shift
=12 complete 256-step episodes. Every one of 3072 transitions receives a fresh
Groth16 proof and admission before execution. All reset boundaries at 64,128,192
must be checked for correct counter continuity. The rule version changes at the
scheduled reset in rule-shift episodes. This checks execution under trusted
reset-state attestation, not a new recursive proof of counter resets.

Paired measurements at 12 fixed indices per episode (protocol.json) add PLONK and
Ed25519 disclosure, including boundaries and the final step. All six protocol
orders are balanced within each episode. 144 paired cases and 3360 total backend
measurements are expected. Every protocol in a paired case uses the same record;
both SNARKs use the same witness and public vector. Generation/verification and
per-call wall latency are distinct; full episode wall latency includes sampled
additional backends and must not be called Groth16-only runtime. State-source and
executor are trusted; the joint prover knows the witness. No policy-inference or
sheaf-energy proof is claimed.

Before the full grid: genuine honest/replay/public-binding tests and in-circuit
negative wall/collision/swap/capacity/quota/state-alias fixtures at populations
2,4,8 for both SNARK protocols. Unexpected library failures do not count as
successful attacks rejected. Unsafe test witnesses never execute.

## Execution and retention

All changes use a new directory and virtual environment. Source/configuration
hashes freeze the full plan. Every run has separate logs/checkpoints/CSV schemas.
The smoke workflow exercises both environments, every method, end-to-end real
cryptography, and analysis. Training is never gated on favorable pilot rewards.
A failed case stops full completion; interrupted attempts are retained. No run is
silently shortened or substituted to meet the submission deadline. Missing full
study results remain unreported, rather than imputed from a smoke test. Archive
all source, configuration, raw results and reference hashes after completeness.

## References and implementation provenance

- RWARE: https://github.com/uoe-agents/robotic-warehouse ; actual installed version
  rware==2.0.0, Gymnasium 1.2.1. The package source hashes are saved separately.
- Gu et al., Multi-Agent Constrained Policy Optimisation, arXiv:2110.02793v2.
  This motivates the adaptive constrained comparator, not a claim of reproducing
  its original continuous-control implementation.
- Existing cryptographic primitives and circuits are reused without modification
  from the checksum-verified PCI_AAMAS27_Rebuild_v1 archive.

### Measurement clarifications
RWARE interventions count institutional action replacements; the frozen custom simulator also counts physical conflict replacements. These fields are reported separately by environment and are not pooled as identical measures. Certificate rows report shared_record_binding_ms once per record (repeated for paired backends for traceability); never sum that repeated quantity across backends. SNARK wall_total_ms includes the individual proof/verification/admission/artifact-write call but excludes shared record construction. Full episode wall time includes all paired calls and is not attributed to one protocol.
