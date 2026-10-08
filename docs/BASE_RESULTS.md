# Completed base study: results and audit scope

The completed archive is `PCI_deadline_results.tar.gz`, SHA-256:

```text
bc7044ba38edea7fb580c51a1228e56ea403ed18091a09a9e86b374c7f88d381
```

Completion: **2026-10-08T01:18:20Z**. The full archive is not uploaded in this repository update. The CSVs below are audited presentation extracts with explicitly limited precision, not raw episode-level evidence.

## Coverage

| Item | Recorded count |
|---|---:|
| Learned runs / controller evaluations | 110 / 12 |
| Joint training steps | 28,835,840 |
| PPO update rounds / optimizer steps | 28,160 / 901,120 |
| Final-policy/controller evaluation episodes | 35,136 |
| Checkpoint-evaluation episodes | 4,400 |
| Certified transition cases / backend measurements | 96 / 288 |
| SNARK proofs in the paired benchmark | 192 |

The completed raw archive contains all methods and coefficient-sweep outcomes. Its full-precision statistics were independently reconstructed from the 122 original per-run evaluation files, without silently skipping malformed rows. Exported aggregates and paired intervals matched to tolerance 1e-12. All 3,447 manifest-listed archive members were rehashed. An independent scalar oracle checked all 244 archived nominal admitted trajectories (62,464 transitions); selected cross-platform replay covered 24 episodes from six two-door seed-zero policies.

These were bounded audits, not exhaustive replay of all policies or independent re-verification of every saved SNARK proof. The base preflight used 21 checks, with adversarial fixtures concentrated on two-agent circuits; larger populations had honest integration measurements. The follow-up expands that coverage.

## Nominal results without institutional admission

Each cell is deliveries per 1,000 joint steps / proposed compliance (%). The shared physical conflict resolver remains active for all methods.

| Layout | Penalty PPO | Full-action consensus | PCI |
|---|---:|---:|---:|
| Open | 97.23 / 88.98 | 92.25 / 90.87 | 91.57 / 90.82 |
| Two doors | 71.94 / 90.80 | 64.97 / 92.40 | 66.02 / 91.64 |
| Staggered | 57.88 / 90.58 | 53.35 / 90.10 | 51.01 / 92.09 |

See the [complete nominal table](../results/base_study_20261008/nominal_unshielded.csv) for all ten methods, including the higher-throughput planning baselines.

PCI minus full-action-consensus throughput has the following paired, pointwise 95% bootstrap intervals:

| Layout | Difference | Interval |
|---|---:|---:|
| Open | -0.68 | [-4.40, 3.71] |
| Two doors | 1.04 | [-4.65, 6.80] |
| Staggered | -2.34 | [-5.60, 2.34] |

These intervals do not establish a general throughput advantage, nor prove equivalence. On staggered, proposed compliance improves by 1.99 percentage points [0.78, 3.35] relative to consensus; that exploratory observation must be read alongside throughput and other layouts. Five training seeds remain five independent training repetitions.

The [paired effects](../results/base_study_20261008/paired_nominal_effects.csv) and [coefficient sweep](../results/base_study_20261008/coefficient_sweep.csv) retain the inconclusive and unfavorable comparisons. The coefficient sweep is not used to select a new primary result after testing.

## Certificate costs

The [certificate medians](../results/base_study_20261008/crypto_medians.csv) report 32 transitions per protocol/population. Generation includes witness calculation plus proving for SNARKs; Ed25519 generation measures a signature and its verifier also reevaluates the disclosed predicate.

| Agents | Groth16 generation / verification (ms) | PLONK generation / verification (ms) |
|---|---:|---:|
| 2 | 299.32 / 9.50 | 14,200.55 / 11.56 |
| 4 | 317.34 / 11.39 | 14,257.10 / 11.50 |
| 8 | 445.23 / 12.37 | 14,351.69 / 11.32 |

This does not support a sub-2-ms SNARK-generation claim. The 32-step certificate episodes are shorter than the quota window and delivered zero, two, and zero tasks at 2, 4, and 8 agents. They establish bounded transition integration, not sustained certified throughput. Protocol order was fixed in this base measurement; full episode wall time includes all paired backends and is not Groth16-only runtime.

## Measurement interpretation

In the custom simulator, action replacements include physical corrections even when institutional admission is disabled. Actual vertex-collision avoidance is a shared simulator invariant, not PCI's exclusive learned benefit. Perfect executed compliance in the admitted evaluation is evidence for the common admission mechanism in those modeled cases, not proof that the learned PCI proposal was always compliant.

The published follow-up protocol targets these evidential limits. No follow-up outcome is inserted into this completed base table.
