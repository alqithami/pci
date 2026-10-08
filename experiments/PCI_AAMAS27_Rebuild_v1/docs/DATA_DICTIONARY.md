# Evidence layout

Every trained run has an immutable config and source/environment provenance,
TRAIN_COMPLETE.json, checkpoint.pt, initial/four milestone policies, updates.csv,
train_episodes.csv, evaluation.csv, learning.csv and EVAL_COMPLETE.json. Checkpoints
contain optimizer, RNG and simulator state for exact same-version resumption.

- env_steps: joint environment steps across vectorized environments.
- agent_steps: env_steps times the training population.
- deliveries: actual pickup-to-depot completions, not shaped reward.
- throughput_per_1000_steps: 1000*deliveries/episode_len.
- proposed_compliance: fraction of originally proposed joint actions satisfying
  walls/collisions/quotas/capacity, before the physical/admission correction.
- executed_compliance: independent predicate check on actions actually executed.
- collision_attempts: agents whose intended moves were stopped by physical conflict.
- actual_vertex_collisions: simulator invariant, zero for ALL methods.
- quota_violations: agents violating modeled resource-entry quotas at proposal time.
- capacity_excess: aggregate region occupancy above capacity before norm admission.
- interventions: count of individual proposed actions replaced by physical or
  institutional resolution. This includes shared physical rejection for all methods.
- mean_consistency_energy: separately squared typed overlap residuals, actual tensors.
- jain_delivery: Jain index of per-agent completed deliveries; blank if no service.
- max_wait_final: max time since last delivery at episode end, not full tail latency.
- task_return: sum of sparse and shaping reward, never substituted for throughput.
- actor_change_l2: optimizer-induced actor parameter change in an update.
- run_id/train_seed/layout/scenario/shield/coefficient: explicit comparison metadata.

First two shielded nominal evaluation episodes per run are stored in compressed
JSONL traces. All held-out episode aggregates are preserved. The full training-step
feature stream is not stored; this is documented rather than reconstructed later.

crypto_benchmark: every measured transition has its statement, private witness,
expected public ABI, proof, actual public vector, and measurement record. Verification
keys and R1CS/WASM are preserved under crypto/build and included in result archives.
The large public PTAU is omitted from the archive, but its URL and checksum are saved.

Strict CSV parsing rejects shifted columns, repeated headers, missing required
fields, impossible probabilities and duplicate evaluation episodes. It never skips
bad lines or pads/trims an unknown schema. COMPLETE requires all planned evaluations
and measured crypto. CORE_SMOKE_ONLY and INCOMPLETE are not paper-result statuses.
