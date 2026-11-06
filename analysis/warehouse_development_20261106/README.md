# Warehouse development analysis

The [public verification script](../../scripts/verify_warehouse_development.py)
checks the source and published sufficient statistics with Python's standard
library. The [results directory](../../results/warehouse_development_20261106/)
contains the full finite candidate grid, every training seed, all profile
counts, and the exact development selection inputs and output.

`development_report.py` is the audit-aware reporter used on the complete
private export of per-episode summaries, completion records, and source and
launch bindings. It requires NumPy and Matplotlib for vector rendering. The
full export is separate from this repository. The reporter binds the published
replay audit and independently aggregates all 5,760 episode summaries. The
original audit used retained actions, native transition replay, an independent
actor reconstruction, and scalar correction checks. Neither the reporter nor
the public check repeats raw model and trace replay.

`training_diagnostics.py` and `bound_inputs.py` analyze the separate closed
training export. The public case-summary table retains all 144 runs. Gradient
ratios are the recorded mean of defined minibatch ratios, with the auxiliary
coefficient already included in the numerator. Oracle rows measure batched
computational work, not elapsed time. Training diagnostics describe the
optimization process and do not establish task efficacy.

The accompanying test scripts construct artificial inputs and refuse to treat
them as research results. They use no cloud services and launch no training:

```bash
python3 analysis/warehouse_development_20261106/test_development_report.py experiments/PCI_M3_Fair_Training_v1
python3 analysis/warehouse_development_20261106/test_training_diagnostics.py experiments/PCI_M3_Fair_Training_v1
```
