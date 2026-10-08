# Implemented cryptographic contract and limitations

Trusted parties: the simulator/state authority, rule registry and atomic executor.
The joint prover knows the complete synthetic multi-agent witness. External auditors
can verify without disclosure of the private coordinates/counters in the proof's
public interface. This does NOT hide data from the prover or state authority, remove
all centralized trust, establish real sensor truth, or implement malicious-secure
multi-party witness generation. Public low-entropy action/rule commitments are not
claimed to hide actions/rules. The state commitment has a fresh secret field salt.

The modeled admission relation consists of bounded integer equalities/inequalities:
legal one-step displacement, valid old/new floor cells, no vertex conflict or
head-on swap, aggregate resource capacity and per-agent entry quotas. Predicates
are conjoined in one all-agent transition circuit. The circuit does not verify PPO
inference, neural consistency energy, reward, task completion, or learned maps.
It does not certify continuous-time robot trajectories or sub-step collisions.

Public statement: run tag, episode, step, nonce, policy version, state root, action
root, rule root, constant ok=1, and a record digest. Hash preimages are recomputed
inside Circom. The Python host uses the exact circomlibjs construction, no fallback.
The verifier gets the expected statement from the trusted simulator/registry and
checks all public signals against it. Each (run,episode,step) can be admitted once,
even with a new nonce. The executor mutates state only after accepted admission.

Conditional claim: under soundness of the selected SNARK and collision resistance
of the commitment hash, a dishonest witness cannot establish this operational
relation for a different expected state/action/rules, apart from cryptographic
failure. Faithful execution and authentic state are additional necessary assumptions.
Finite negative tests are evidence of implementation checks, not a proof of the
underlying cryptographic assumptions or a comprehensive security audit.

Authenticated-disclosure baseline: genuine Ed25519 signatures plus independent
predicate reevaluation. It exposes the disclosed state. A valid signature alone is
not treated as a compliance proof. This rebuild does not implement Pedersen,
recursive proofs, proof aggregation, streaming accumulators or blockchain.

All saved witnesses are synthetic simulator data. Their retention is intentional
for reproduction. Do not reuse the default archive settings with confidential data.
The local phase-2 ceremony is a benchmark setup with one operator. It is not a
production recommendation. Never deploy these newly authored circuits on funds,
physical systems or sensitive inputs without independent cryptographic review.
