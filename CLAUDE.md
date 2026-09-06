# Training Blueprint

Use [README.md](README.md) for the operator and training adapter.
The manifests and lockfile select SDK APIs; read those sources before changing dependency integration.

## Verification

- For a complete chain-to-training flow, use [lifecycle.rs](operator/tests/lifecycle.rs).
  It requires Docker, seeded chain fixtures, and Python ML dependencies, and is ignored by default.
  Prove the operator trains, writes a checkpoint, evaluates held-out data, and returns the result on-chain.
- For two-operator training, use [two_operator_demo.rs](operator/tests/two_operator_demo.rs).
  It uses real adapters with an in-memory message broker; it does not prove network transport.
- For actual network exchange, use [gossip_sync.rs](operator/tests/gossip_sync.rs).
- Test contract changes with actual deployments under `contracts/`.

Do not count an ignored test or a missing prerequisite as a successful execution.
Preserve independent evaluation data and test nontrivial optimizer, checkpoint, and synchronization behavior.
Select checks for the changed behavior; do not require new tests for documentation edits.
