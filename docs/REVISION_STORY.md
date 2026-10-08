# PCI: revised scientific story and response-to-concerns map

**Evidence cutoff:** the audited base study completed on 8 October 2026. The separate follow-up is not yet incorporated as completed evidence in this document. This is a public development narrative, not a submitted rebuttal or a reproduction of confidential review text.

## Central argument

Agents governed by overlapping institutional contexts face three different questions: whether contextual decisions agree on shared quantities, whether a concrete joint action satisfies operational rules, and what evidence establishes that satisfaction. PCI gives these questions separate computational interfaces. Action-producing context heads are regularized on declared overlap effects; an independent operational predicate checks the candidate transition; a certificate binds that predicate to the expected interaction before admitted execution.

The institution is an explicit versioned specification of contexts, overlap maps, predicates, and an evidence/admission contract. Coordination is the policy or planner operating within that specification. Cryptographic verification is neither the coordination algorithm nor evidence that a low consistency score establishes safety.

A warehouse example makes the separation concrete. Safety and task-efficiency heads can agree on expected displacement while an independently chosen action still contributes to excessive joint doorway occupancy. An access quota is a specified allocation rule, not a theorem of distributive fairness. The institution determines admissibility; task reward determines utility; neither a consistent representation nor a valid signature alone substitutes for checking the relevant predicate.

## What the completed rebuild changes

| Concern | Concrete revision | Evidence and remaining boundary |
|---|---|---|
| Baselines were named without quantitative comparisons | Publish the full nominal table and paired effects for the six learned methods and four specified controllers | Base results retain stronger controllers and null/negative PCI effects. These are not relabelled state-of-the-art reproductions. |
| Mathematical notation and guarantees exceeded the implemented object | Use maps into a common overlap space and separately squared residuals; distinguish compatibility from admissibility | Fixed, state-instantiated maps and gradient tests. No learned-institution, neural-convergence, or first-cohomology claim. |
| The proof system was unspecified | Provide concrete Circom generator, range and commitment constraints, Poseidon host parity, snarkjs interfaces, and admission checks | Actual Groth16/PLONK measurements and negative tests, under trusted state/executor and a joint prover. |
| A single trace or long runtime was treated as validation | Separate optimizer checks, held-out results, statistical reconstruction, trace oracles, and selected policy replay | Audit scope is enumerated below; not every possible trajectory or checkpoint was replayed. |
| Figures did not answer comparative questions | Report deliveries/compliance together, paired differences, intervention costs, and certificate timing distributions | Source analysis generates these from identifiable rows; constant success signals are not decorative performance plots. |
| Results did not match headline claims | Replace unsupported prototype percentages and microsecond SNARK claims with audited values | No 94% completion, 73% reduction, or sub-2-ms proving claim is retained as a result of the new study. |
| Institution, coordination, and verification were conflated | Define their roles and trust assumptions separately | The implementation still has trusted state and execution; no blanket claim of trust-free decentralization. |

## What the evidence currently supports

The base study contains 110 trained runs, 12 controller evaluations, five independent training seeds, and three custom layouts. Its raw-data audit reconstructed all 35,136 final-policy/controller evaluation rows and the original paired bootstrap intervals. A separately written oracle checked all 244 saved nominal admitted traces, totaling 62,464 transitions. Selected cross-platform replay covered 24 episodes. These checks establish the stated execution and measurement consistency, not universal algorithmic validity.

PCI's base learning results show a compliance–throughput trade-off relative to scalar penalties. Its throughput differences from full-action consensus have pointwise 95% intervals containing zero on all three layouts. That is not proof of equivalence, and it is not evidence of a general throughput advantage. The shuffled-map comparisons likewise do not isolate a robust throughput benefit from the original semantic projections. The paper should explain these results rather than conceal them behind one favorable metric.

The base certificate study establishes real, bounded transition certification: 96 transitions, 288 paired backend measurements, and 192 SNARK proofs. Median Groth16 generation is approximately 299–445 ms and PLONK generation approximately 14.2–14.4 seconds in that workload. Verification is cheaper, but synchronous admission must account for generation as well. Thirty-two-step episodes do not exercise the 64-step quota reset. Full production reliability, rare-event timing tails, and certified long-horizon task completion are not established by that sample.

The [base results](BASE_RESULTS.md), [CSV provenance](../results/base_study_20261008/README.md), and [cryptographic contract](../experiments/PCI_AAMAS27_Rebuild_v1/docs/THREAT_MODEL.md) preserve these qualifications.

## Why the follow-up is a separate study

The follow-up was designed **after** inspection of the base outcomes. It does not retroactively strengthen the base preregistration, replace unfavorable results, or select a winning original coefficient. Its fixed [protocol](../experiments/PCI_AAMAS27_Followup_v1/docs/PROTOCOL.md) and [machine-readable plan](../experiments/PCI_AAMAS27_Followup_v1/protocol.json) define three targeted questions.

**Mechanism:** does semantic overlap structure matter after operator-scale control? Trace-normalized PCI is compared with normalized full-action consensus and an orthogonal nonsemantic action-space rotation preserving each centered map's spectrum, rank, and norm. Realized gradient norms are logged because operator normalization does not guarantee equal policy-gradient strength. Normalized PCI is a new variant, not the original objective renamed.

**Comparator and environment:** does the observed effect persist against an empirical-rate adaptive PPO-Lagrangian implementation and within an independently implemented warehouse environment? The comparator has reward/cost critics and a projected multiplier; it is an adaptation, not an exact reproduction of a named published method. RWARE retains native dynamics but receives documented common features, shaping, and institutions. This is independent implementation evidence within warehousing, not cross-domain generality or an unmodified RWARE benchmark result.

**Execution coverage:** do existing frozen policies admit full 256-step certified episodes spanning quota resets and a rule-version change? The plan adds multiple checkpoints, populations, and regimes, with balanced protocol ordering on paired samples. It reuses the custom-warehouse relation; no RWARE proof, recursive episode proof, or policy-inference certificate is claimed.

All follow-up sample sizes remain planned until the complete artifacts and analysis are audited. No provisional result is included here.

## Theory and positioning

At a fixed state, a positive-weight sum of squared overlap residuals is zero exactly when each residual is zero. A distance-to-kernel bound depends on the smallest positive singular value of the weighted operator. These are mathematical interpretations of compatibility, not a proof that neural PPO converges globally, that operational constraints are feasible, or that task return improves. A separate affine contradiction diagnostic has a known oracle; it does not diagnose institutional infeasibility from high neural training loss.

The contribution must be assessed against existing institutional/norm-enforcement, sheaf-coordination, constrained-learning, and proof-system literature. The implementation uses established cryptographic primitives. The repository does not claim to execute SIGMA, Sheaf-ADMM, MACPO, or the published MAPPO-Lagrangian code; mentioning those methods in the paper is not an empirical comparison. Claims of priority or state-of-the-art superiority require a separate literature and implementation assessment.

## Reporting rule after the follow-up

Keep original and follow-up objectives, seeds, environments, endpoints, and timing contracts distinct. Report each configuration and all fixed comparisons, including failures and zero-delivery episodes. Analyze task throughput together with proposal compliance, cost rates, and intervention costs. Average held-out episodes within trained seeds before estimating training uncertainty. State when an interval is pointwise/exploratory; do not convert absence of a detected difference into equivalence.

A favorable new result can support the specific new mechanism and workload tested. An inconclusive or negative result limits that claim. Successful certificates support the operational evidence contract under its assumptions, regardless of whether PCI beats a coordination baseline. The final narrative must preserve this separation.
