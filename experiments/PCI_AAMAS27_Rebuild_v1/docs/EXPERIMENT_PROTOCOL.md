# PCI rebuild: preregistered execution protocol (v1)

## Relation to the supplied manuscripts
This is a NEW implementation, not a reconstruction of the unavailable earlier code.
The journal manuscript describes a 10 x 10, three-robot prototype and reports only
its authenticated-logging backend. No old number, trace, or claim is imported here.
The new benchmark uses documented 13 x 13 layouts with four training agents. The
restriction maps are fixed constructions instantiated from the current state;
they are not learned institutional parameters. The formal object tested is a
cellular sheaf on the context graph (the 1-skeleton), not a novel H^1 obstruction.

## Questions decided before examining full-run results
Q1: Does typed, action-grounded compatibility improve held-out delivery/compliance
tradeoffs over equally parameterized scalar-penalty PPO and ordinary consensus?
Q2: Does it reduce interventions when every method receives the same admission
monitor? A perfect shielded compliance rate alone is not evidence for PCI learning.
Q3: How do fixed policies transfer to tighter quotas, a rule change, and 2/8/12 agents?
Q4: Can concrete operational transitions be certified by real Groth16/PLONK with
correct binding, range checks, and rejection of deliberately invalid witnesses?
Q5: What are the measured costs of the compatibility operator and of actual proofs?

## Simulator, observations, and rewards
PCI-Warehouse is a new, discrete, synchronous grid simulator, not Overcooked, RWARE,
a digital twin, a physical robot deployment, or a previously validated public benchmark.
All three layouts are connected 13 x 13 grids with a one-cell wall border. Pickups
are at (2,2), (2,4), (2,8), (2,10); depots at (10,3), (10,9). The source specifies
all walls and two controlled regions. Agents must reach a pickup, then a delivery
zone, before an actual delivery is counted. Each agent has an exogenously seeded
sequence of tasks. Episode duration is 256 joint steps; counters reset every 64.
Normal region capacity is 2 and each agent may enter each region 3 times/window.
Quota-tight evaluation uses 1; rule-shift evaluation reduces 3 to 2 at step 128.
This is a modeled access quota, not a theorem of social or distributive fairness.

All controllers receive the same collision-resolving physical dynamics. Vertex
conflicts and head-on swaps are stopped, so actual vertex collisions should be
zero even without PCI. Attempted conflicts and institutional violations are logged
separately. The optional admission monitor additionally prevents capacity/quota
violations and logs its interventions. Reporting prevented collisions as collisions
or attributing the simulator's collision avoidance to PCI is prohibited.

For agent i: task reward = delivered + 0.2*pickup + 0.05*(old_distance-new_distance)
- 0.01*movement. The cooperative reward is the mean across agents. Except for the
task-only baseline, it subtracts 0.25 times the mean violation cost (wall attempt,
physical-conflict rejection, quota violation, and capacity excess divided by N).
Delivered items, not shaped rewards, define throughput: 1000*deliveries/joint_steps.

Each candidate action has 23 features explicitly listed in environment.py.
Context heads see fixed masks; all receive local task direction. The pooled-state
actor additionally sees the mean feature tensor across agents. It is NOT an
optimal or fully informative centralized upper bound. All actors have the same
number of parameters; all training critics have the same centralized pooled input.

## Policy / consistency formulation
Three action-scoring heads (safety, efficiency, allocation) each contain two
64-unit tanh layers. The executed categorical policy uses their averaged logits.
Every head therefore affects actions; there is no detached decorative summary head.
The compatibility term compares expected displacement/occupancy, region entries,
or progress/delivery effects at the three corresponding context overlaps. At a
fixed state, these are linear maps of local action probabilities. Their construction
is deterministic but their numerical coefficients are state-conditioned.

Ordinary consensus compares the full 5-action distributions. PCI-shuffled changes
the action correspondence on one end of every overlap to [2,4,1,0,3]. This is a
negative-control structural ablation, not a physically meaningful alternative rule.
The norm of each residual is squared BEFORE aggregation. All regularizers are
PyTorch tensors and participate in backpropagation. No random metric fallbacks exist.

PPO: gamma .99, GAE .95, clip .2, entropy .02, value coefficient .5, norm clip .5,
Adam learning rate .0003, four update epochs, early stopping at approximate KL .03.
Eight environments, rollout 128, minibatches of 128 environment-time groups; the
agent dimension is preserved. Counters identify joint environment and agent steps
separately. Complete model/optimizer/RNG/environment checkpoints are resumable.

## Deadline profile: exactly 110 learned runs + 12 controller evaluations
Main matrix: 6 methods x 3 layouts x 5 independent training seeds = 90.
Methods: task-only PPO; scalar-penalty PPO; ordinary consensus; PCI; shuffled-map
PCI; pooled-state PPO. Main lambda=.2. Each run has 262,144 joint environment steps.
Matched coefficient sweep: PCI and consensus on twodoors, lambda in {.05,1}, 5
seeds = 20 additional runs. Total training steps = 28,835,840 joint steps.
This is not hyperparameter tuning followed by selective reporting: report the
whole sweep, including negative results, with main lambda fixed beforehand.

Four explicitly implemented non-learning controllers on each layout: greedy rules,
reservation auction, two-round broadcast consensus, and a centralized recursive
priority planner. These are operational baselines, not faithful reproductions of
unavailable old implementations, a certified optimizer, or a theorem-backed MPC
upper bound. Do not claim their algorithms reproduce PIBT's entire specification.

All learned runs: 24 held-out episodes for nominal, quota-tight, rule-shift, scale2,
scale8, scale12, both with and without admission. Held-out environment seeds are
100000..100023, disjoint from training. Controllers use those same episodes but
are not duplicated and misrepresented as five independent training seeds.
Initial and 25/50/75/100% checkpoints also get eight separate held-out episodes
(200000..200007) for learning curves. The last checkpoint, not the best test score,
is the primary model. Population transfer is across episodes, NOT dynamic churn.

## Cryptographic measurement, separate from expensive policy learning
Real proof preflight runs BEFORE the training grid. It checks Poseidon parity,
honest proofs, replay, public tampering, private-state substitution, unsigned range
aliases, walls, vertex collisions, swaps, aggregate capacity and quotas. Both
Groth16 and PLONK must pass. Missing libraries, missing keys, or inconclusive error
messages do not count as rejection tests passing.

The prespecified cryptographic anchor is PCI/twodoors/lambda .2/seed 0. Its frozen
policy proposes actions in fresh, seed-controlled episodes with N=2,4,8. For each
of 32 steps/population, the monitor constructs an admissible joint transition,
Groth16 proves it and admission verifies BEFORE execution. PLONK and disclosed
Ed25519+monitor evaluate that same transition. The resulting 96 transitions and
288 measurements are a bounded integration benchmark, not a throughput claim
for every training step or a large production workload. All valid and invalid
self-test outcomes remain in the archive.

Circuits recompute movement, obstacle membership, collisions, head-on swaps,
all-agent region occupancy, quotas, range constraints, and commitments. The
prover cannot substitute a Python-generated collision flag. Unsafe transitions
are unsatisfiable, not valid proofs with ok=0. Roots bind the actual action,
private state, geometry, counters, active rules, session/episode/step, and nonce.

The public phase-1 transcript comes from the URL and BLAKE2b checksum pinned in
crypto_build.py from the official snarkjs README. Groth16 uses an additional
single-operator OS-random phase-2 contribution: suitable for this explicit
benchmark trust model, not a claim of a production multi-party ceremony.

## Statistical plan and reporting
Learned methods: aggregate evaluation episodes within each training seed first;
then equally weight training seeds. Use a 5000-resample percentile bootstrap for
pointwise 95% intervals. With one seed, no interval is estimated. Small-seed
intervals are uncertain; they are not proof of generalization to all domains.
Controllers: episode-level intervals are labelled separately, since they have
no training population. The primary paired contrasts compare PCI to penalties,
consensus, shuffled maps and the pooled-state actor within matching seeds.
Contrasts are exploratory pointwise intervals, not multiplicity-corrected tests.

A Pareto chart uses genuinely distinct method/coefficient conditions, actual
deliveries, and proposed-action compliance without admission. Its frontier is
empirical nondominance of observed means, not a confidence-qualified global optimum.
Perfect admitted compliance is summarized as an invariant/check; it does not
create an informative all-ones scatter plot. No fabricated threshold is fitted.

Proof timings are repeated measurements, not independent training seeds. Report
median, p95 and sample counts. Witness+proof generation is one measured operation;
verification excludes key JSON loading after cache warmup. Wall total includes
host binding, RPC, artifact writes and (for Groth16 admission) a second verification.
Ed25519 size is raw signature bytes; SNARK size is JSON serialization bytes. Their
encodings differ and must not be described as equally compressed proof formats.

Solver diagnostic: affine cyclic equalities with specified gap have analytic
minimum gap^2/n; least squares and an independent LP feasibility check corroborate
it. This is explicitly a synthetic consistency diagnostic, not warehouse evidence
or a claim that high neural training loss proves institutional infeasibility.

## Deadline / outcome rules
No claim of positive PCI lift is made before results. No increased circuit size or
artificial delay is added merely to make the run slow. Missing data cannot be
silently dropped, merged into the last CSV field, or assigned invented run IDs.
The script refuses mixed configurations and changed-source resumption. A crash
retains checkpoints, logs and the failure report. Partial analysis must be requested
explicitly and is marked INCOMPLETE. A smoke run is never the paper evaluation.
