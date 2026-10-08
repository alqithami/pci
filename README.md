# Proof-Carrying Institutions (PCI) for Distributed AI

Research code for studying **contextual compatibility, operational admissibility, and verifiable evidence as distinct requirements** in multi-agent systems.

PCI compares action-producing contextual policies on their shared, institutionally relevant effects. A separate admission contract checks concrete operational predicates and binds a certificate to the expected state, action, rule version, and interaction. Low compatibility loss is not a safety certificate; a valid proof is not evidence that the learning mechanism outperforms its baselines.

## Study status — audited 8 October 2026

| Study | Evidence status | Scope |
|---|---|---|
| **Base rebuild** | Completed and independently audited | 110 learned runs, 12 controller evaluations, three custom layouts and five training seeds; 96 certified transitions and 288 backend measurements. |
| **Post-audit follow-up** | Completed; statistical reconstruction and saved-SNARK verification passed | 125 learned runs, 36,000 final-policy evaluations, documented RWARE extension and normalized mechanism controls; 12 complete certified episodes, 3,072 transitions and 180 deliveries. |

**Findings are condition-dependent, not uniform superiority.** The base study does not establish a general throughput advantage over consensus or shuffled maps. The follow-up finds a throughput gain over normalized consensus and the spectrum-matched nonsemantic control on RWARE tiny, but not over fixed penalties. Custom two doors exhibits a throughput/compliance trade-off; RWARE small does not establish a throughput gain. Nominal RWARE records no quota/capacity violations, making the separate tighter-quota evaluation important. See [completed follow-up findings and audit boundaries](docs/FOLLOWUP_RESULTS.md).

Every one of the follow-up's 3,216 saved SNARK proofs was independently reverified. These finite checks and full modeled episodes do not establish physical-world security, production reliability, or universal learning benefits. Both studies and their scientific source fingerprints remain separate and unchanged.

## Repository map

- [`experiments/PCI_AAMAS27_Rebuild_v1`](experiments/PCI_AAMAS27_Rebuild_v1): custom simulator, PPO variants, controllers, circuits, cryptographic interfaces, tests, analysis, and packaging.
- [`experiments/PCI_AAMAS27_Followup_v1`](experiments/PCI_AAMAS27_Followup_v1): separate post-audit study, importing the intact base code and its frozen checkpoints/circuit artifacts for certification.
- [`results/base_study_20261008`](results/base_study_20261008): audited base presentation extracts, not full raw evidence.
- [`results/followup_20261008`](results/followup_20261008): audited follow-up presentation extracts and audit counts; full raw archive remains author-held.
- [`docs/REVISION_STORY.md`](docs/REVISION_STORY.md): updated scientific narrative and concern-to-evidence map.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): installation, tests, fresh runs, frozen-archive reuse and completion checks.
- [`docs/BASE_RESULTS.md`](docs/BASE_RESULTS.md) and [`docs/FOLLOWUP_RESULTS.md`](docs/FOLLOWUP_RESULTS.md): separate measured outcomes and limitations.
- [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md): archive identities and availability.

## Start with a source check, not a full training run

```bash
git clone https://github.com/alqithami/pci.git
cd pci
python3 scripts/verify_sources.py
```

The standard-library verifier checks manifest-listed scientific source files and detects unlisted files in exported source directories. It does not train models, assess scientific claims or verify proofs. See the reproduction guide for the 57 base and 29 follow-up Python tests and the separate genuine cryptographic preflight. Cloning and source verification never start the expensive suites.

## What is implemented

**Base learning:** task-only PPO, fixed penalties, full-action consensus, PCI typed overlaps, shuffled maps and a parameter-matched pooled-state actor. Four additional controllers implement specified rule, reservation-auction, broadcast-consensus and centralized-priority procedures. They are custom operational comparators, not optimality upper bounds or wholesale reproductions of named published algorithms.

**Follow-up learning:** fixed penalties, normalized consensus, normalized PCI, spectrum-matched nonsemantic projections, and an empirical-rate PPO-Lagrangian adaptation with reward/cost critics. The new objective and comparator do not retroactively replace base outcomes. PPO-Lagrangian here is not an exact MACPO/MAPPO-Lagrangian reproduction.

**Certificates:** genuine Groth16 and PLONK transition proofs, plus measured Ed25519 authenticated disclosure with predicate reevaluation. Circuits recompute movement, floor membership, collisions, swaps, aggregate capacity, quotas, ranges and commitments. Expected public signals and replay prevention are checked before admitted execution. Ed25519 signature bytes were not archived for independent post-run replay, unlike the saved SNARK proofs.

## Boundaries that matter

Both environment families remain warehousing. RWARE retains native dynamics with documented common task/map features, shaping and institutions; these are not unmodified leaderboard scores. The custom circuit does not certify RWARE. The simulator/state authority and executor are trusted. The joint prover sees the witness; privacy is relative to an external auditor.

No physical sensor authenticity, continuous-motion safety, malicious-secure multiparty proving, policy-inference proofs, learned institutional maps, higher-order cohomological detection, dynamic churn or recursive proofs are claimed. Read the [threat model](experiments/PCI_AAMAS27_Rebuild_v1/docs/THREAT_MODEL.md) and [follow-up protocol](experiments/PCI_AAMAS27_Followup_v1/docs/PROTOCOL.md).

## Provenance and licensing

Scientific sources are preserved; repository-level summaries/manifests are packaging additions. Presentation CSVs are rounded extracts. Full archives, checkpoints, installed lockfiles and heavy cryptographic artifacts are not uploaded as release assets in this update; their identities and current availability are explicit in ARTIFACTS.md.

The historical base protocol uses “preregistered” in its heading. This release describes it as prespecified; no independent registry entry is supplied. The follow-up was designed after inspecting base outcomes and remains explicitly post-audit.

No new license is assigned. See [NOTICE.md](NOTICE.md). Manuscript drafts, confidential reviews, credentials, private connection settings and operational account logs are excluded. This author-identified repository is not an anonymized review artifact.
