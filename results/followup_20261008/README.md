# Audited follow-up presentation extracts

Source: the completed post-audit follow-up archive `PCI_followup_results.tar.gz`, SHA-256 `45f52edf232c6162fba4d6e3168ef02e4c12cb0abd53c46a4b73b5f8f9998708`. The archive finished on 8 October 2026 and its 18,663 payloads were independently checked. See [the full interpretation](../../docs/FOLLOWUP_RESULTS.md).

These CSVs are **rounded presentation extracts**, not the full-precision per-episode files. Nominal means retain all 25 method/configuration combinations. Primary paired effects retain all four PCI-minus-baseline comparisons at all five configurations for throughput and proposed compliance. No favorable-only comparator selection is used. The original archive remains the primary evidence and has not been uploaded in this update.

`throughput` is actual deliveries per 1,000 joint steps. `proposed_compliance` is a fraction; multiply by 100 for percentages or percentage-point differences. Each learned estimate averages 24 episodes within each of five training seeds. The paired pointwise intervals use the original 5,000 bootstrap resamples with seed 92170; they are exploratory and not multiplicity-adjusted.

The certificate table uses the balanced-order paired subset: 48 records per population per protocol. SNARK generation includes witness and proof; the signature baseline discloses and reevaluates the record. Do not substitute all-transition Groth16 medians or call whole-episode wall time Groth16-only time.

Important: nominal RWARE evaluations have no logged quota/capacity violations; tighter-quota evidence must be considered separately. The RWARE and custom-world intervention definitions differ. The normalized follow-up objectives and new training seeds must not be pooled with the base study as unchanged replications.
