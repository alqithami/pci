# Evidence-to-paper integration checklist

This document prepares the next paper revision without inventing outcomes or altering the submitted manuscript. It maps the frozen protocol and actual output schema to the arguments they can support. The running study's results remain pending until the full completion and audit gates pass.

## Distinct evidence sets

| Evidence set | Identity and use | Do not infer |
|---|---|---|
| Base learning/certificate study | 110 policies; original commitments; archived base results | That its timings measure the later repair |
| Completed learning follow-up | 125 new policies; normalized controls and RWARE; original-protocol full episodes | General learning superiority or a non-warehouse domain |
| Later nominal sensitivity analysis | Exact existing counts; 40 nominal tests; original bootstrap retained plus t/sign-flip/Holm sensitivity | A retroactive confirmatory design or new training replication |
| Blinded-action functional validation | New two-agent circuit/keys; separate opening tests | Full-episode benchmark completion or a privacy theorem |
| Running cross-policy certificate study | New blinded keys; 15 frozen checkpoints; 270 planned episodes | New learned policies, GPU acceleration, recursion, or results available at original submission |

The original and normalized learned objectives are different. The running cross-policy experiment uses the original base methods `pci`, `ppo_penalty` and `consensus`, not the normalized follow-up models. Its inference remains over five archived training seeds after averaging two matched evaluation episodes.

## What can be written now

The institutional interface, threat-model roles, authenticated-input requirement, private-action-opening construction, rule-version semantics, replay contract, exact metric denominators and fixed experimental design are specified in the released code. State clearly that authentic state and faithful execution are assumed; the single-host benchmark does not implement sensor attestation, PKI or a crash-persistent distributed transaction ledger.

Distinguish a signed assertion from an authenticated disclosed record and a relation proof. A valid signature does not establish the truth of an assertion by an untrusted signer. The disclosed comparator verifies the signature, reconstructs bindings and reevaluates the predicate. The SNARK checks the encoded relation for the expected input commitments. All interfaces require correctly anchored verification keys. Functional examples of false signed assertions are not cryptographic forgeries.

The old action commitment can be enumerated over small public action spaces. The new domain-1004 action commitment uses a private random field opening, not a public nonce as a hiding secret. The full private research archive and disclosed comparator intentionally reveal openings; they must not be mixed into a claim about the SNARK-only auditor view. Auxiliary-information and side-channel privacy require a separate argument.

## Outputs required before writing the new results

| Paper question | Exact output | Required interpretation/check |
|---|---|---|
| What changes between policy families under the same gate? | `results/full/episode_summary.csv`; `results/full/analysis/seed_means.csv` | Average both evaluation episodes within each trained seed; retain every population/regime/method and zero-delivery outcome. |
| Is useful throughput traded against institutional corrections? | `results/full/analysis/paired_operational_comparisons.csv` | Show throughput and additional institutional replacements together; 36 primary contrasts with separately adjusted test families, not the earlier 40-test learning family. |
| How much is the proposal changed? | Per-case `trace.jsonl.gz` and `episode.json` | Separate proposed-to-physical correction, physical-to-admitted correction, individual action components, and joint steps affected. |
| What are the evidence-interface costs? | `results/full/analysis/paired_backend_summary.csv`; per-case `measurements.csv` | Use the matched paired subset and balanced order. Generation, verification, shared binding and call wall times are distinct; repeated binding costs must not be summed four times. |
| Did the model execute what was certified? | Per-case `AUDIT.json`, public proofs, private witnesses and traces | Verify expected public inputs, retained signatures and proof validity, transition predicates, state continuity and quota resets. Internal same-library verification is not an external security proof. |
| Does the full episode finish with valid evidence? | `results/full/SUITE_COMPLETE.json`, `results/full/analysis/ANALYSIS_COMPLETE.json`, final archive manifest and checksum | Require the complete fixed matrix and all gates, not individual `EPISODE_COMPLETE` lines. |

Expected complete counts are 270 episodes, 69,120 proof-gated transitions, 6,480 paired records, 88,560 backend calls and 75,600 saved SNARK proofs. These remain design targets until checked. Every transition is Groth16-gated; the other three backends are paired measurements at selected steps, not separate closed-loop execution treatments.

## Analysis and presentation guardrails

Use the final complete dataset rather than interim means. Keep the original bootstrap values visible and explain test assumptions and small-seed uncertainty. Do not tune the protocol, stop early for significance, replace a seed or choose a checkpoint because its outcome looks favorable. Repeated timings within a rollout are not independent training replications. An absent detected difference is not equivalence; any degenerate zero-variance test must be interpreted with its actual seed values, not as certainty.

Whole-episode wall time includes the sampled extra backends and trace work. The current summary is written before per-case replay; post-case verification adds further workflow time. Do not label the summary as Groth16-only deployment latency or compare it with old shared-host timings as a controlled acceleration experiment. Report the CPU affinity and host load, and distinguish proof-library verification from end-to-end validation.

The signed-assertion keys are generated in the benchmark; a deployed auditor would need an authentic signer-key registry. The theorem assumptions must include correct predicate encoding and circuit/key selection. A fresh action opening is an implementation repair, not by itself a theorem of complete trajectory confidentiality.

## Completion-to-revision handoff

After the final marker and checksum pass, preserve the complete archive and verify all manifest members. Independently reconstruct the operational seed means and the 36 contrasts, check denominators and all strong comparators, and inspect the retained failures/interrupted attempts before drafting conclusions. Add final source, checkpoint and circuit identities to any released presentation tables. Only then update the next manuscript and repository results with a dated post-submission label.

A fresh confirmatory learning study, multiple map draws, official constrained-learning reproductions, train-with-monitor training, a non-warehouse domain and stronger compatibility-to-action theory remain separate tasks. They are not completed by this certificate workflow or this repository update.
