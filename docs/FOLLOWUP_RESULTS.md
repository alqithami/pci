# Completed PCI follow-up: audited outcomes

The separate post-audit follow-up finished at **2026-10-08T08:46:26Z**. This release records completed evidence, not planned sample sizes. The original base study and all versioned scientific sources remain unchanged.

## Archive and independent checks

`PCI_followup_results.tar.gz`, 532,707,227 bytes, SHA-256:

```
45f52edf232c6162fba4d6e3168ef02e4c12cb0abd53c46a4b73b5f8f9998708
```

The server checksum and the copied archive checksum agree. A separate streaming checker rehashed all **18,663 manifest-listed payloads**, with no hash mismatches, missing/unlisted payloads, duplicate member names, or unsafe paths. The full raw archive remains author-held; it is not uploaded as a GitHub release asset in this update.

An independently written reconstruction checked all per-run configurations, checkpoints, evaluation schemas and seed coverage, actual-delivery arithmetic, per-agent totals, finite metrics and nonzero actor updates. Run means and all **1,680 paired intervals and 2,100 condition intervals** reproduce the original export to absolute tolerance 1e-12. The original bootstrap settings remain 5,000 draws, seed 92170, and five independent training seeds.

| Evidence | Verified count |
|---|---:|
| Learned runs / held-out policy evaluations | 125 / 125 |
| Joint training steps | 32,768,000 |
| Update rounds / optimizer steps | 32,000 / 1,024,000 |
| Final-policy / checkpoint-evaluation episodes | 36,000 / 5,000 |
| Zero-delivery evaluation episodes retained | 1,945 |
| Certified episodes / transitions | 12 / 3,072 |
| Backend measurements / saved SNARK proofs | 3,360 / 3,216 |
| Certified delivery events | 180 |
| Counter resets / rule-version changes checked | 36 / 6 |

Every saved Groth16 and PLONK proof was independently reverified with the unchanged keys. A separate implementation recomputed Poseidon state/action/rule bindings and checked the public ABI against the expected record. An independent scalar checker assessed geometry, movements, collisions, swaps, aggregate capacities, quotas and state continuity on all 3,072 certified transitions. Six fresh public-input tamper checks were rejected.

This is not a new training replication, exhaustive policy replay, or a cryptographic security proof. Ed25519 signature bytes were not archived and were not independently replayed; their timing results remain producer-recorded measurements with archive integrity.

## Primary learning findings

Nominal four-agent evaluation has institutional admission disabled. Throughput counts actual deliveries per 1,000 joint steps, not shaping reward. Each mean averages episodes within each of five training seeds before averaging seeds. The data directory retains every method and configuration.

| Configuration | Normalized PCI minus normalized consensus throughput | Pointwise 95% paired interval |
|---|---:|---:|
| Custom open | -1.82 | [-7.49, 4.49] |
| Custom two doors | -4.07 | [-7.52, -0.94] |
| Custom staggered | 4.56 | [-1.73, 8.50] |
| RWARE tiny | 2.67 | [1.76, 3.58] |
| RWARE small | -0.36 | [-1.46, 0.75] |

On RWARE tiny, normalized PCI obtains 13.80 deliveries/1,000 steps, compared with 11.13 for normalized consensus and 12.14 for the spectrum-matched nonsemantic control. Its paired gains are 2.67 [1.76, 3.58] and 1.66 [0.23, 3.32]. The contrast against fixed penalties is 0.16 [-2.25, 1.76], so improvement over that simpler comparator is not established. Nominal compliance is 99.834% for PCI and 99.893% for consensus; the favorable throughput comparison is not universal dominance.

Custom two doors shows a trade-off: PCI loses 4.07 [-7.52, -0.94] deliveries relative to consensus while gaining 1.66 percentage points [0.15, 2.80] of proposed compliance. RWARE small does not establish a throughput advantage over any comparator. Its unfavorable compliance comparison against fixed penalties is also retained.

The adaptive PPO-Lagrangian comparator has higher compliance and lower throughput on the custom layouts. Its mean cost rates are 0.0137, 0.0196 and 0.0126 (open/two doors/staggered), compared with PCI's 0.0428, 0.0365 and 0.0409. The configured cost target is 0.02. Across-seed means are not per-iteration guarantees; more throughput at greater cost is not dominance over the constrained objective.

All intervals are exploratory and pointwise, not multiplicity-adjusted. Five seeds remain five independent training repetitions. An interval containing zero is not an equivalence result.

## Constraint exposure is an essential qualification

All **1,200 nominal unshielded RWARE episodes** have zero logged quota violations and capacity excess, and their admitted counterparts have zero institutional replacements. Near-perfect nominal compliance is therefore not evidence of resolving active quota/capacity violations. The metric also includes physical wall/conflict attempts.

The tighter-quota scenario produces **6,624 unshielded agent-step quota violations** across the RWARE methods/layouts. This is an exposure count, not 6,624 episodes. Under tight quotas on tiny, PCI's throughput differences are 2.15 [1.17, 2.99] versus consensus and 1.43 [0.29, 2.54] versus the rotation control. Against fixed penalties, throughput is -0.36 [-2.47, 1.27] and compliance improves by 1.70 percentage points [0.94, 2.45]. Small again does not establish throughput gains.

RWARE retains native dynamics with explicitly added common task/map features, shaping and institutional rules. It is independent implementation evidence within warehousing, not a second real-world domain or unmodified benchmark score. RWARE limits region entries; the custom simulator limits standing occupancy. Intervention fields have different scopes and are not pooled as identical measures.

## Complete certified episodes

Frozen base PCI checkpoints at seeds 0 and 1 execute twelve 256-step episodes across 2/4/8 agents and nominal/rule-shift regimes. Every transition receives a fresh Groth16 proof before execution. All episodes deliver tasks, 6–25 each and 180 total, with modeled executed compliance one. All 36 quota resets and six rule-version changes are checked.

At 144 fixed paired records, each backend measures the same record and all six protocol orders are balanced within each episode. Use the paired subset (48 records per population per backend) for protocol comparisons:

| Agents | Groth16 generation / verification (ms) | PLONK generation / verification (ms) |
|---|---:|---:|
| 2 | 296.07 / 8.72 | 14,265.75 / 11.59 |
| 4 | 316.52 / 9.32 | 14,247.57 / 11.46 |
| 8 | 450.41 / 12.44 | 14,364.07 / 11.69 |

Generation includes witness computation and proving for SNARKs. Ed25519 signs a disclosed record and reevaluates its predicate; it has different functionality. The larger all-transition Groth16 sample has slightly different medians and is not mixed with the paired protocol table. Full episode wall time includes sampled additional backends and is not Groth16-only time.

This is stronger modeled-episode integration evidence than the base 32-step sequences. It retains trusted state/executor and joint-prover assumptions. There is no RWARE circuit, policy-inference proof, sensor-authenticity guarantee, multiparty private proving, recursive episode proof, or high-frequency real-time claim.

## Release scope and conclusion

The [follow-up data](../results/followup_20261008/) are rounded presentation extracts, not full raw data. Complete full-precision audit tables and helpers remain with the verified author-held artifacts. The source code and original follow-up protocol are unchanged.

The supported narrative is **condition-dependent benefits of selective compatibility, together with independently checked full-episode institutional execution**. The results do not establish uniform learning superiority. The original base study remains separate and visible; further theory, references and manuscript-readiness review must not be inferred from successful experiment execution.
