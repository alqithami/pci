# Audited presentation extracts — completed base study

Source: the 8 October 2026 independent audit of `PCI_deadline_results.tar.gz` (SHA-256 `bc7044ba38edea7fb580c51a1228e56ea403ed18091a09a9e86b374c7f88d381`). These four CSVs are copied byte-for-byte from the supplied audited-results presentation package.

**They are not the complete raw results or full-precision analysis.** Nominal means are displayed to three decimals, paired contrasts and the coefficient sweep to six decimals, and crypto medians retain their displayed precision. Use original per-run data and archived analysis for exact numerical reconstruction.

- `nominal_unshielded.csv`: all ten methods, main coefficient 0.2, four agents, nominal rules, no institutional admission. Throughput counts actual deliveries per 1,000 joint steps; compliance is a percentage. Learned estimates average 24 evaluation episodes within each of five training seeds. Custom controllers have evaluation-episode rather than training-seed replication.
- `paired_nominal_effects.csv`: PCI minus each named baseline. Compliance differences are fractions, not percentages; multiply by 100 for percentage points. Intervals resample five paired seed means 5,000 times with random seed 9217. They are exploratory, pointwise 95% intervals, not multiplicity-adjusted tests.
- `coefficient_sweep.csv`: two-door layout, PCI and consensus, coefficients 0.05, 0.2, and 1.0. All listed outcomes are retained; no best-test coefficient is promoted.
- `crypto_medians.csv`: repeated measurements on the same paired transition cases. Signature sizes are raw bytes; SNARK sizes are compact snarkjs JSON bytes. Pretty-printed archive files can be larger. Size encodings and cryptographic/disclosure guarantees differ.

Custom-world action replacements include shared collision/wall correction; do not interpret them as exclusively institutional rejection. The RWARE follow-up's intervention measure has a different scope and must not be pooled with these values as though identical.

Full artifact availability and verification are described in [ARTIFACTS.md](../../docs/ARTIFACTS.md). No incomplete follow-up data is included.
