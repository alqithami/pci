# Frozen fair-training study source

This is the 43-file source snapshot used for the completed November 2026
warehouse development cohort. The canonical source-map SHA-256 is
`e7fbf20f649ce40a21d42d644756cf519dd6fd2717564c5bbcf0eb0793473ed7`.
`protocol/PROTOCOL_DRAFT.json` is the original design body; its historical
filename and draft status are preserved. Scientific execution used a separate
source-bound design and launch record on the qualified host.

`runtime/training/` contains the four methods and environment adapters;
`runtime/evaluation_fair/` contains the retained final-policy evaluation and
audit. `runtime/cohort_fair.py` is the bounded controller. The older evaluation
modules in this exact source closure are retained for source identity and are
not the evaluator used for this study.

Start with the [public analysis check](../../scripts/verify_warehouse_development.py).
This source snapshot does not include trained checkpoints, raw traces, host
access details, or a reusable scientific launch authorization. The runtime
requires a newly qualified environment and separately bound inputs before a
new scientific run. Native RWARE additionally needs its original reference
qualification material.
