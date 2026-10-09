# Completed learning follow-up: outcomes and statistical sensitivity

The learning follow-up finished at **2026-10-08T08:46:26Z**. Its observations and source are unchanged. This page now incorporates the later statistical and public-input interpretation corrections; it contains no outcomes from the running blinded-action cross-policy benchmark.

## Archive and check scope

`PCI_followup_results.tar.gz`, 532,707,227 bytes, SHA-256:

```
45f52edf232c6162fba4d6e3168ef02e4c12cb0abd53c46a4b73b5f8f9998708
```

The server and copied archive checksums agree. All 18,663 manifest-listed payloads passed a separate streaming hash check. A separate reconstruction checked per-run configurations, checkpoint identities, evaluation schemas, seed coverage, actual-delivery arithmetic, per-agent totals, finite metrics and actor updates. The original 1,680 paired and 2,100 condition intervals reproduced to absolute tolerance 1e-12 using the recorded 5,000 bootstrap draws and seed 92170.

| Evidence | Count |
|---|---:|
| Learned policies / final-policy episodes | 125 / 36,000 |
| Joint training steps | 32,768,000 |
| Update rounds / optimizer steps | 32,000 / 1,024,000 |
| Checkpoint-evaluation episodes | 5,000 |
| Zero-delivery evaluation episodes retained | 1,945 |
| Original-protocol certified episodes / transitions | 12 / 3,072 |
| Backend measurements / saved SNARK proofs | 3,360 / 3,216 |
| Certified delivery events | 180 |
| Counter resets / rule-version changes checked | 36 / 6 |

All saved Groth16 and PLONK proofs were reverified with unchanged keys and separately recomputed bindings. A scalar checker assessed movement, collisions, aggregate capacities, quotas and state continuity over all certified transitions. Six fresh public-input tamper checks failed as expected. These are internal artifact checks using the same cryptographic libraries, not independent retraining or a formal security audit. Ed25519 signature bytes were not retained for independent replay.

## Original nominal outcomes remain visible

Institutional admission is disabled in these nominal four-agent learning comparisons; shared physical conflict resolution remains active. Throughput counts actual deliveries per 1,000 joint steps. Evaluation episodes first average within each trained seed.

| Configuration | PCI minus normalized consensus throughput | Original pointwise bootstrap interval |
|---|---:|---:|
| Custom open | -1.82 | [-7.49, 4.49] |
| Custom two doors | -4.07 | [-7.52, -0.94] |
| Custom staggered | 4.56 | [-1.73, 8.50] |
| RWARE tiny | 2.67 | [1.76, 3.58] |
| RWARE small | -0.36 | [-1.46, 0.75] |

RWARE-tiny mean throughput is PCI 13.80, consensus 11.13, rotation 12.14, fixed penalty 13.64 and adaptive PPO-Lagrangian 15.07. PCI versus fixed penalties is 0.16 [-2.25, 1.76] under the original bootstrap. Custom two doors trades lower throughput for 1.66 [0.15, 2.80] percentage points of higher compliance. RWARE-small does not establish a throughput advantage. These measured differences are not uniform superiority, and an interval containing zero is not equivalence.

The adaptive comparator has higher compliance and lower throughput on the custom layouts. On RWARE it has higher mean tiny throughput but lower joint proposal compliance; across-seed composite cost means are about 0.0144/0.0159 on tiny/small. Its target 0.02 is an averaged composite agent-step cost, not a 2% joint-action failure bound. The recorded RWARE multipliers did not saturate the cap 10. Means below the target do not guarantee satisfaction for every seed or update.

## Complete-family sensitivity changes the strength of inference

The [reproducible reanalysis](../analysis/nominal_sensitivity_20261009/) uses exact nominal counts for all 125 policies, representing 3,000 held-out episodes. It preserves the original bootstrap and adds paired-t intervals, exhaustive sign-flip tests and separate Holm corrections across five configurations, four comparators and two endpoints: **40 comparisons**.

Neither adjusted family contains p<0.05. On RWARE-tiny, the rotation throughput difference 1.66 has paired-t interval [-0.75, 4.07], compared with bootstrap [0.23, 3.32]. The consensus difference 2.67 has paired-t interval [1.21, 4.12], unadjusted t p=0.0070, exhaustive sign-flip p=0.0625, and Holm-adjusted t p=0.2592. The favorable original intervals therefore remain exploratory rather than confirmatory evidence of semantic advantage.

Paired-t and sign-flip procedures require different assumptions; sharing a seed establishes neither normal differences nor exchangeability. Five seeds, not 3,000 episodes or 5,000 bootstrap resamples, determine training replication. The one fixed rotation and unequal realized gradient strength remain attribution limits. The analysis is a later sensitivity check, not a retroactive preregistration.

## Rule exposure

All 1,200 nominal unshielded RWARE episodes have zero recorded quota violations and capacity excess, with zero additional institutional replacements in their admitted counterparts. Near-perfect nominal proposal compliance chiefly reflects the remaining physical-attempt metric, not resolution of active quota/capacity conflicts.

Tight-quota RWARE instead records 6,624 unshielded agent-step quota violations. On tiny, the original throughput differences are 2.15 [1.17, 2.99] versus consensus and 1.43 [0.29, 2.54] versus rotation. Against fixed penalties, throughput is -0.36 [-2.47, 1.27] and compliance improves by 1.70 [0.94, 2.45] percentage points. These are secondary exploratory outcomes, not substitute primary tests.

RWARE uses native dynamics plus documented common task/map features, shaping and institutions. It is not an unmodified benchmark or a non-warehouse domain. RWARE limits entries while the custom simulator limits standing occupancy; intervention fields should not be pooled as identical measures.

## Original-protocol certified episodes and privacy boundary

Frozen base PCI checkpoints 0/1 execute twelve complete 256-step episodes at 2/4/8 agents under nominal and rule-shift conditions. Each transition receives Groth16 admission. Episodes deliver 6–25 tasks each, totaling 180, with modeled executed compliance one. These are transition proofs across episodes, not recursive episode proofs.

On 144 fixed paired records, the three backends are measured in balanced six-way order. There are 48 paired samples per population/backend:

| Agents | Groth16 generation / verification (ms) | PLONK generation / verification (ms) |
|---|---:|---:|
| 2 | 296.07 / 8.72 | 14,265.75 / 11.59 |
| 4 | 316.52 / 9.32 | 14,247.57 / 11.46 |
| 8 | 450.41 / 12.44 | 14,364.07 / 11.69 |

SNARK generation includes witness computation and proving. Signature generation and disclosed predicate reevaluation supply different assurance/disclosure. Episode wall time includes sampled additional backends and is not Groth16-only time. No high-frequency real-time claim follows.

Original replacements count agent decisions, including physical correction: at eight agents, 155–176 of 2,048 decisions are replaced (7.57–8.59%). The separately counted 74–88 affected joint steps represent 28.91–34.38% of the episode. [Case-level denominators are included.](../analysis/nominal_sensitivity_20261009/data/reanalysis/intervention_units.csv)

**These measurements do not establish trajectory confidentiality.** The original zero-salt action commitments are enumerable over small public action spaces. Known initial state and recovered movement can expose subsequent positions. This is leakage from the public statement, not a break of SNARK zero knowledge. The later private-action-opening repair needs its own keys, tests, timing and privacy analysis. The current [post-submission benchmark](POSTSUBMISSION_REPRODUCIBILITY.md) is separate and pending; none of its planned totals replace this table.

## Availability

The existing [follow-up CSVs](../results/followup_20261008/) remain unchanged rounded presentation extracts. Exact sufficient statistics and unchanged sensitivity scripts are now available for the complete nominal family. Full original episode rows, checkpoints, proof archives and transitive dependency locks remain author-held, with identities in [ARTIFACTS.md](ARTIFACTS.md). No public raw-archive URL is claimed.
