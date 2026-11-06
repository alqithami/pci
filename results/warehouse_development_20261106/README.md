# Audited warehouse development results

All 144 planned policies completed 262,144 joint training steps each on an
NVIDIA L4 on IBM Cloud. Each final policy was evaluated on ten matched worlds
in each of four trained profiles. The complete evaluation contains 5,760
episodes, 1,474,560 joint steps, and 5,898,240 agent decisions. The retained
record audit passed all episodes, with no replacement policy rollouts. The
reporting pipeline independently reconciled the saved case summaries and
recomputed the original development selection.

| Family | Selected candidates | Hypervolume of candidate mean points | Mean of seed-level selected-pair hypervolumes |
| --- | --- | ---: | ---: |
| Fixed penalty | `fixed_05`, `fixed_10` | 0.015667 | 0.015632 |
| Direct risk | `direct_02`, `direct_11` | 0.016685 | 0.016772 |
| PPO-Lagrangian | `lagrangian_03`, `lagrangian_11` | 0.015548 | 0.015562 |
| Shield-aware | `shieldaware_00`, `shieldaware_04` | 0.025244 | 0.025244 |

The selector uses the third column with the fixed cost bound of 0.05
institutional replacements per agent decision and utility normalized by agent
decisions. The fourth column is a different, descriptive calculation; it did
not change the selection. The shield's zero correction rate is structural. Its
best candidate delivered 100.98 tasks per 1,000 joint steps and requested 625
additional policy-oracle rows per joint step. The common admission checker is
outside that query count. The count does not measure latency or proof savings.
The second shield point is redundant for the mean-point hypervolume and follows
the fixed tie rule.

The Direct pair uses nonzero auxiliary weights. At its low-correction point,
`direct_11` averaged 1.123 institutional replacements per 1,000 agent decisions
and 62.04 deliveries per 1,000 joint steps. The corresponding low-correction
Fixed point, `fixed_10`, averaged 1.457 and 49.09. These points were selected on
the development data. Three training seeds are three independent clusters;
episodes and profiles are repeated observations. No confirmatory test,
noninferiority claim, or proof-work result is made here.

## Reproduce the published summaries

From the repository root, run `python3 scripts/verify_warehouse_development.py`.
It verifies the released file hashes, all 48 candidates, the 144 seed rows and
576 profile rows, the oracle accounting, and the frozen selection. It does not
read raw traces or repeat the original retained-action audit.

- [All candidate means](candidate_means.csv), [seed counts](candidate_seed_points.csv), and [profile counts](profile_seed_counts.csv)
- [Deployment oracle work](deployment_oracle_work.csv) and [training diagnostic summaries](training_case_summary.csv)
- [Seed-level hypervolumes](frozen_selected_pair_hypervolumes.csv) and [exact selection receipt](DEVELOPMENT_SELECTION.json)
- [Audit and transfer scope](EVIDENCE_SUMMARY.json) and [cohort completion record](STUDY_COMPLETE.json)
- [All candidate trade-offs](all_candidate_tradeoffs.svg), [Direct grid](direct_beta_lambda_grid.svg), [Fixed sweep](fixed_beta_sweep.svg), and [profile trade-offs](per_profile_tradeoffs.svg)

The fresh confirmation design keeps these eight configurations and uses 20 new
training-seed clusters. The controlled execution study and native RWARE
confirmation are separate requirements for the integrated family of tests.
