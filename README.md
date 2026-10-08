# Proof-Carrying Institutions (PCI) for Distributed AI

Research code for studying **contextual compatibility, operational admissibility, and verifiable evidence as distinct requirements** in multi-agent systems.

PCI compares action-producing contextual policies on their shared, institutionally relevant effects. A separate admission contract checks concrete operational predicates and binds a certificate to the expected state, action, rule version, and interaction. Low compatibility loss is not a safety certificate; a valid proof is not evidence that the learning mechanism outperforms its baselines.

## Study status — 8 October 2026

| Study | Evidence status | Scope |
|---|---|---|
| **Base rebuild** | Completed and independently audited | 110 learned runs, 12 controller evaluations, three custom warehouse layouts, five training seeds; 96 certified transitions and 288 paired backend measurements. |
| **Post-audit follow-up** | Running at the last check; full results not released or audited here | 125 planned runs, normalized mechanism controls, an adaptive PPO-Lagrangian comparator, a documented RWARE extension, and longer certified episodes. |

**The completed base study does not establish a general throughput advantage for PCI over full-action consensus or shuffled maps.** It demonstrates genuine learning, a compliance–throughput trade-off, and functioning bounded transition certification. These findings remain visible rather than being replaced by the follow-up.

This repository update is a source/documentation release. It does not restart or alter the ongoing experiments. Smoke success, test counts, and planned experiment sizes are not presented as completed follow-up results.

## Repository map

- [`experiments/PCI_AAMAS27_Rebuild_v1`](experiments/PCI_AAMAS27_Rebuild_v1): versioned custom simulator, PPO variants, controllers, Circom generator, Poseidon/snarkjs worker, tests, analysis, and packaging.
- [`experiments/PCI_AAMAS27_Followup_v1`](experiments/PCI_AAMAS27_Followup_v1): separate post-audit study; depends on the intact base code and, for certification, base checkpoints and cryptographic artifacts.
- [`results/base_study_20261008`](results/base_study_20261008): audited **presentation extracts**, not the full raw-data archive.
- [`docs/REVISION_STORY.md`](docs/REVISION_STORY.md): the updated scientific narrative and concern-to-evidence map.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): installation, tests, fresh runs, frozen-archive reuse, completion checks, and limitations.
- [`docs/BASE_RESULTS.md`](docs/BASE_RESULTS.md): completed outcomes and audit scope.
- [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md): archive identity and which evidence is or is not included in Git.

## Start with a source check, not a full training run

```bash
git clone https://github.com/alqithami/pci.git
cd pci
python3 scripts/verify_sources.py
```

The verifier uses only Python's standard library. It checks every manifest-listed versioned source file and detects unlisted files in the exported source directories. It does not certify the mathematics, train a model, or verify SNARK proofs.

Use the [reproduction guide](docs/REPRODUCIBILITY.md) to install the appropriate dependencies and run the 57 base Python tests or 29 follow-up Python tests. Cryptographic integration is a separate, explicit check. The full suites consume substantial compute and are never launched by cloning or by the source verifier.

## What is implemented

**Base learning:** task-only PPO, fixed-penalty PPO, full-action consensus, typed-overlap PCI, shuffled-map PCI, and a parameter-matched pooled-state actor. Four additional controllers implement specified rule, reservation-auction, broadcast-consensus, and centralized-priority procedures. These custom controllers are not claimed to reproduce entire published algorithms or provide optimality upper bounds.

**Follow-up learning:** fixed penalties, trace-normalized consensus, normalized PCI, a spectrum-matched nonsemantic projection control, and empirical-rate PPO-Lagrangian with reward/cost critics. The normalized objective and adaptive comparator are new follow-up variants, not retrospective replacements for the base study. The comparator is a documented adaptation, not an exact MACPO/MAPPO-Lagrangian reproduction.

**Certificates:** genuine Groth16 and PLONK operational-transition proofs, plus Ed25519 authenticated disclosure with predicate reevaluation. Circuits recompute movement, floor membership, collisions, swaps, aggregate capacity, quotas, ranges, and commitments. Expected public signals and replay prevention are checked before admitted execution.

## Boundaries that matter

The custom environment and RWARE remain warehouse task families. RWARE uses native dynamics with explicitly added common task/map features, shaping, and institutions; these are not unmodified RWARE leaderboard scores. The custom circuit does **not** certify RWARE.

The simulator/state authority and executor are trusted. The joint prover sees the multi-agent witness; privacy is relative to an external auditor. No claim is made of physical sensor authenticity, continuous-motion safety, malicious-secure multiparty proving, policy-inference proofs, learned institutional maps, higher-order cohomological detection, dynamic churn, or recursive proofs.

Detailed [threat model](experiments/PCI_AAMAS27_Rebuild_v1/docs/THREAT_MODEL.md) and [follow-up protocol](experiments/PCI_AAMAS27_Followup_v1/docs/PROTOCOL.md) accompany the source.

## Provenance and licensing

The experiment directories preserve the supplied versioned scientific source bytes. Repository-level guidance and manifests are packaging additions. Installation-generated dependency locks and heavy artifacts are not reconstructed from memory: see [artifact availability](docs/ARTIFACTS.md).

The original protocol uses the word “preregistered” in its historical heading. This release describes it as **prespecified**; no independent registry entry is supplied. The follow-up was specified after inspecting the base outcomes and is explicitly post-audit.

No new license is assigned by this update. See [NOTICE.md](NOTICE.md). Manuscript drafts, review exports, credentials, host-specific connection instructions, and private operational logs are not included.
