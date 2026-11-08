PCI estimator-mechanism analysis — prospective software release
9 November 2026

This is a separate analysis for the mechanism study already declared in
protocol/build_study.py. The original warehouse/RWARE/execution Holm12 family
and its frozen modules are unchanged. This directory does not authorize a
scientific launch and contains no scientific estimator outcomes.

Scope
-----
The two development-selected nonzero-auxiliary configurations are direct_02
(beta=0, lambda=0.2) and direct_11 (beta=1, lambda=0.8). Each is trained with
exact, score/leave-one-out, and conditional cost gradients on ten fresh
initialization clusters, 64001–64010: 60 policies, 524,288 joint steps each,
and 31,457,280 training joint steps in total. Sampled estimators use K=32
and every minibatch state. All four profiles have equal weights. Evaluation
uses twenty worlds per profile, making 4,800 episodes and 1,228,800 joint steps.

The endpoint is the normalized two-policy dominated hypervolume within each
training cluster, with cost and delivery utility per agent decision, a fixed
cost bound of 0.05, and the same two base configurations for all three modes.
Two-sided Exact-minus-Score and Exact-minus-Conditional tests enumerate all
2^10 sign assignments with inclusive ties and exact arithmetic. These two
contrasts form their own complete Holm2 family at alpha=0.05. Whole-cluster
bootstrap intervals and paired-t sensitivity intervals remain pointwise.
Zero paired variance leaves paired-t and dz undefined. Nonsignificance is
neither equivalence nor noninferiority. All points, including those above the
area bound, are retained. No selection changes are made from test outcomes.

The detailed numerical recipe was clarified after the separate warehouse
confirmation and before opening any scientific outcome from seeds 64001–64010.
The prior protocol declared this estimator study, candidates, budgets, and
separate two-test family. There is no claim of an independent registry entry.

Qualification
-------------
Required runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0.

Run before scientific outcomes are opened:
  python -B qualify_estimator_analysis.py --source-root /absolute/frozen/source --receipt-tag UNIQUE_TAG

The qualifier runs synthetic statistics, full-shaped receipt, and pipeline
controls. Its canonical lock remains QUALIFICATION_IN_PROGRESS_NOT_LAUNCH
until all controls pass; candidate pipeline tests run in an isolated copy.
The resulting PASS_SYNTHETIC_QUALIFICATION lock is still subject to independent
source/hash review and actual CPU/CUDA training readiness. It is not launch
authority, a proof of simulator fidelity, or third-party data attestation.

Source/design requirements
--------------------------
The 43 executable source files and protocol must match the original frozen
source map. The design lock additionally binds:
  stage: estimator_ablation
  study_domain: warehouse
  training_policies: 60
  training_seeds: [64001, ..., 64010]
  development_selection_sha256: original selection body hash
  estimator_analysis_recipe_sha256: this exact recipe body hash
  estimator_analysis_implementation_lock_sha256: final reviewed software lock

CUDA_READINESS.json must include:
  estimator_modes_pass: {exact: true, score: true, conditional: true}

Root review must establish those booleans from actual source-bound engineering
receipts for the relevant estimator shapes, RNG/checkpoint recovery and native
oracle integration. Existing exact-only qualification cannot be relabeled as
sampled-estimator qualification. Source/configuration/launch/readiness, deadline,
host/device, and execution-lock bindings must all agree.

Analysis command after a complete audited cohort
-----------------------------------------------
  python -B estimator_analysis.py --cohort /absolute/closed/cohort --source-root /absolute/frozen/source --selection /absolute/DEVELOPMENT_SELECTION.json --case-manifest /absolute/inputs/cases/CASE_MANIFEST.json --out /absolute/NEW_RESULT.json

The adapter reconstructs all 60 prescribed configurations and 4,800 case
descriptors with the estimator-stage evaluation namespace. It checks every
summary against its receipt, the retained replay-audit bindings and complete
file membership, then independently aggregates profiles and policy counts.
It rejects synthetic markers in production mode. Result files cannot be
overwritten. The adapter checks source-bound case summaries and original
published audit receipts; it does not replay raw models/traces again or
authenticate an external publisher. Data-origin and physical validity remain
outside its guarantee.

No scientific result can be claimed from the synthetic fixture packets,
software checks, or independent NumPy formula controls.
