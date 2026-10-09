# Reproduce the complete nominal statistical sensitivity analysis

This is a reanalysis of the completed 125-policy follow-up, not data from the running post-submission certificate benchmark. The two scripts are unchanged from the final technical supplement. The exact count values are unchanged; JSON whitespace is compacted for the repository.

## Inputs and units

`data/reanalysis/nominal_counts.json` contains 25 configuration/method cells. Each row is `[training_seed, total_deliveries, compliant_joint_steps]` across that trained policy's 24 nominal unshielded evaluation episodes. Every episode lasts 256 joint steps, so the denominator per policy is 6,144. There are 125 policies and 3,000 represented episodes. The canonical JSON SHA-256 is:

```
91c68f761cdc483beddf1622b36e70e91fca91314557126af0a25baf77531524
```

These event totals were previously checked against all 125 original evaluation CSVs and their recorded hashes. This repository includes those sufficient statistics for the two primary nominal endpoints, not the original episode rows or all secondary outcomes. The archived source remains unchanged.

`data/followup/primary_paired_effects.csv` is the exact existing rounded bootstrap extract, duplicated only to preserve the original checking script's paths. `data/reanalysis/intervention_units.csv` counts individual action replacements and affected joint timesteps separately for the twelve completed original-protocol certified episodes. It is not an outcome table for the new circuit.

## Run without training or proof generation

Use Python with NumPy and SciPy. The recorded analysis environment pins NumPy 2.3.5 and SciPy 1.17.0 in the experiment requirements. From the repository root:

```bash
python3 analysis/nominal_sensitivity_20261009/checks/reanalyze_nominal.py
python3 analysis/nominal_sensitivity_20261009/checks/check_sensitivity.py
```

The first command generates `nominal_seed_means.csv` (125 rows), `nominal_paired_seeds.csv` (200 rows), `nominal_sensitivity.csv` (40 rows), and a report under `data/reanalysis/`. The second runs 14 calculation/identity checks, including agreement with SciPy's exhaustive sign-flip procedure and preservation of the original bootstrap values to their stored precision. Generated outputs are ignored by Git; all inputs needed to reproduce them are included. Neither command connects to the experimental server.

## Interpretation

The family is five configurations times four comparators times two nominal endpoints. The original 5,000-draw percentile bootstrap (seed 92170) is retained. Paired-t intervals and exhaustive two-sided sign flips are sensitivity procedures, with Holm correction applied separately to their 40-test families. Neither adjusted family contains a p-value below 0.05. This is not a retroactive confirmatory design and is not an equivalence finding.

On RWARE-tiny throughput, PCI minus the fixed rotation is 1.66015625 deliveries per 1,000 steps. Its paired-t interval is approximately [-0.754, 4.074], versus the original bootstrap [0.228, 3.320]. PCI minus consensus is approximately 2.669, with unadjusted t p=0.0070 but sign-flip p=0.0625 and Holm-adjusted t p=0.2592. Favorable original pointwise intervals should not be described as established general semantic superiority.

Exact paired-t coverage requires independent normal differences; exact sign flipping needs its sign-symmetry or label-exchangeability null. Shared random seeds alone establish neither assumption. Five seeds remain five independent training replications. Integer arithmetic is used for the sign-flip statistic to avoid floating-point changes to equality comparisons.

The script defines explicit conventions for zero empirical variance, including a zero p-value for a nonzero constant difference. Such a degenerate case is not evidence that five repetitions establish population certainty; inspect seed values and model assumptions. No test method should be selected after looking for a favorable answer.
