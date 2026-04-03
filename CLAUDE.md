
## Testing Standards

**Priority order (highest to lowest):**
1. **Full lifecycle tests** using `SeededTangleTestnet` — anvil + Tangle Core contracts + `BlueprintRunner` + `TangleProducer/Consumer`. Job submitted on-chain → operator processes → result submitted on-chain. This is the only test that proves the system works.
2. **Real server integration tests** — start the actual axum server, mock only the external backend (the GPU process we cannot run without hardware), send real HTTP requests, verify real responses.
3. **Real contract tests** — `forge test` with actual Solidity logic: registration validation, pricing, payment splitting, access control.
4. **Real algorithm tests** — test actual math/logic (DeMo optimizer, layer range calculation, checkpoint hashing). Only where the logic is non-trivial.

**What is NOT acceptable:**
- Serialization roundtrip tests (these prove nothing)
- Mocking our own code (mock the external dependency, not our server)
- Tests that pass with empty/hardcoded responses
- "Coming soon" or stub tests that test nothing
- Unit tests of getters/setters

**Testing tools:**
- `SeededTangleTestnet` / `MultiHarness` from `blueprint-anvil-testing-utils` for full lifecycle
- `wiremock` for mocking external backends (vLLM, diffusion, embedding servers)
- `forge test` with real contract deployment for Solidity
- `anvil` for local EVM testing
- Real `blueprint-manager` binary for operator lifecycle

**Every PR must include:**
- A test that exercises the actual user flow end-to-end
- No decrease in test coverage of critical paths
- Contract tests for any new on-chain logic


## Blueprint SDK Testing Tools (use these)

The Blueprint SDK provides real testing infrastructure at `blueprint-anvil-testing-utils`:

```rust
// Full lifecycle test — the gold standard
use blueprint_anvil_testing_utils::{MultiHarness, OperatorFleet, OperatorSpec};

let harness = MultiHarness::builder()
    .add_blueprint("my-blueprint", my_router(), service_id)
    .spawn()
    .await?;

let handle = harness.handle("my-blueprint").unwrap();

// Submit a real on-chain job
let submission = handle.submit_job(JOB_INDEX, payload).await?;

// Wait for operator to process and return result on-chain
let result = handle.wait_for_job_result(submission).await?;

// Decode and verify
let decoded: MyResultType = MyResultType::abi_decode(&result)?;
assert_eq!(decoded.status, "success");
```

**Available tools:**
- `SeededTangleTestnet` — boots anvil with all Tangle Core contracts
- `MultiHarness` — multi-blueprint test harness with operator fleets
- `BlueprintHandle` — submit jobs, wait for results, check status
- `TestRunner` — lightweight single-blueprint runner
- `OperatorFleet` — configure honest/malicious operators for Byzantine testing
- `TangleHarness` — lower-level harness with direct contract access

**Dev dependencies to add:**
```toml
[dev-dependencies]
blueprint-sdk = { git = "https://github.com/tangle-network/blueprint", branch = "main", features = ["testing", "tangle"] }
blueprint-anvil-testing-utils = { git = "https://github.com/tangle-network/blueprint", branch = "main" }
wiremock = "0.6"
tempfile = "3"
```

