# Proof-Carrying Institutions (PCI) for Distributed AI

Research code for **institutional specification, operational admission, and interaction-bound evidence**, with contextual policy regularization as an optional proposal mechanism.

An institution declares the rules; a policy proposes an action; a common monitor can correct it; a certificate establishes the encoded relation for the expected admitted action. Low compatibility loss is not a safety certificate, and successful verification is not evidence of superior coordination or general trajectory confidentiality.

## Evidence status — 9 October 2026

| Study | Status | Scope |
|---|---|---|
| **Base rebuild** | Completed; archived and checked | 110 learned runs, 12 controller evaluations, three custom layouts; 96 certified transitions and 288 backend measurements. |
| **Learning follow-up** | Completed; archived and checked | 125 learned runs, 36,000 final-policy episodes, RWARE extension and normalized controls; twelve original-protocol certified episodes, 3,072 transitions and 180 deliveries. |
| **Blinded-action functional validation** | Two-agent functional checks passed; not a full benchmark | Separate private action-opening circuit and new keys; 21 real-circuit checks plus ten opening/binding checks. |
| **Post-submission cross-policy benchmark** | **Running; full results pending** | 270 planned complete episodes comparing frozen PCI, fixed-penalty and consensus policies with a separately keyed blinded-action protocol. |

The dated [execution snapshot](docs/POSTSUBMISSION_STATUS.json) is not a live monitor. The ongoing experiment is not modified by this repository update. Its planned counts must not be described as completed outcomes, and it does not train new policies or launch CUDA training.

**Learning findings remain exploratory.** The complete 40-comparison nominal sensitivity analysis retains the original bootstrap results and adds paired-t/sign-flip procedures with separate Holm corrections. Neither corrected test family contains p<0.05. The favorable RWARE-tiny rotation contrast is sensitive to interval choice; fixed penalties are not generally beaten. This limits inference rather than establishing equivalence. [Reproduce the primary nominal analysis without retraining.](analysis/nominal_sensitivity_20261009/)

**The original protocol does not establish trajectory confidentiality.** Its public zero-salt action commitments can be enumerated for small action spaces. The later private-opening circuit is a separate variant: old episode timings and proofs are not measurements for that repair. Functional binding tests and random openings alone are not an application-level privacy theorem.

## Repository map

- [Base experiment](experiments/PCI_AAMAS27_Rebuild_v1/): simulator, learned methods, controllers, circuits, tests, analysis and packaging.
- [Completed learning follow-up](experiments/PCI_AAMAS27_Followup_v1/): normalized mechanisms, nonsemantic control, adaptive PPO-Lagrangian, RWARE adapter and original-protocol full episodes.
- [Blinded-action preparer and functional tests](experiments/PCI_Blinded_Action_Prototype_v2/): four preserved scripts that create a separate circuit variant; no frozen original source is edited.
- [Running post-submission experiment](experiments/PCI_Postsubmission_Certified_v1/): exact launched source, fixed protocol, host tests, CPU profiler, staged validation, cross-policy execution, analysis and archive verification.
- [Nominal sensitivity analysis](analysis/nominal_sensitivity_20261009/): exact event totals for 125 policies, unchanged analysis/check scripts, and separate intervention units.
- [Base presentation results](results/base_study_20261008/) and [follow-up presentation results](results/followup_20261008/): historical rounded extracts retained unchanged.
- [Scientific narrative and evidence map](docs/REVISION_STORY.md), [post-submission reproduction](docs/POSTSUBMISSION_REPRODUCIBILITY.md), and [paper-integration checklist](docs/PAPER_EVIDENCE_MAP.md).
- [Original reproduction guide](docs/REPRODUCIBILITY.md) and [archive identities](docs/ARTIFACTS.md).

## Safe first checks

```bash
git clone https://github.com/alqithami/pci.git
cd pci
python3 scripts/verify_sources.py
python3 scripts/verify_postsubmission_sources.py
```

Both checks use the Python standard library and start no training, proof generation, remote connection, or deployment. The original verifier covers the 52 base/follow-up source files; the new verifier covers the benchmark, prototype scripts, and sensitivity inputs. Read the relevant reproduction guide before invoking expensive workflows. **Do not pull repository changes into a running experiment directory or start another copy of its controller.**

## What the new benchmark tests

Three policy families, five archived seeds, three populations (2/4/8), three rule regimes, and two evaluation seeds give 270 planned 256-step episodes. Each admitted step requires Groth16. At 24 fixed points per episode, Groth16, PLONK, a signed assertion, and authenticated disclosure are measured in balanced order. The full targets are 69,120 proof-gated transitions, 6,480 paired records, 88,560 backend measurements, and 75,600 saved SNARK proofs.

The protocol separates physical action replacements, additional institutional replacements, and affected joint timesteps. Signatures and their public keys are retained, while SNARK-only public views are separated from synthetic private openings and disclosure views. Comparisons use five previously trained seeds, not a new 20–30-seed confirmatory learning study. Complete-data analysis and archive checks must finish before interpreting results.

## Scope, trust and preservation

The learning implementations use fixed within-agent action-effect maps. The adaptive comparator is a documented empirical-rate PPO-Lagrangian adaptation, not an exact MACPO/MAPPO-Lagrangian reproduction. RWARE retains native dynamics with common added features, shaping and rules; these are not unmodified benchmark scores or a separate real-world domain. The custom circuit does not certify RWARE or neural-policy inference.

Authentic expected input commitments, correct circuit/key selection, and faithful execution remain necessary. A signed assertion can suffice when the auditor trusts the signer's predicate evaluation; a signature by itself authenticates a claim, not its truth. The benchmark's signature helpers use generated keys and do not implement a deployment PKI. SNARK-only disclosure must not be confused with the full research archive, which intentionally retains synthetic witnesses/openings. No sensor attestation, malicious-secure multiparty proving, crash-persistent transaction protocol, recursive proof, general privacy theorem, or production-security audit is claimed.

Scientific source snapshots and existing result extracts are preserved. Full checkpoints, original raw archives, installed dependency locks and proving keys remain separate; no public raw-archive download is invented. The historical protocol heading uses “preregistered”; the repository describes that study as prespecified because no independent registry entry is supplied. The later studies and reanalyses are explicitly separate.

No new license is assigned; see [NOTICE.md](NOTICE.md). Confidential reviews, manuscript drafts, conversation/assistance logs, credentials, private connection details, raw witnesses, checkpoints, and live operational logs are not published in this update. This author-identified repository is not an anonymized review artifact.
