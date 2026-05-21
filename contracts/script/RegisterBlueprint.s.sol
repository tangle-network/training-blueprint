// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { DistributedTrainingBSM } from "../src/DistributedTrainingBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys `DistributedTrainingBSM` and registers the distributed-training
///         blueprint on Tangle in a single broadcast.
///
///         `DistributedTrainingBSM` is a regular (non-upgradeable) contract whose
///         constructor takes the Tangle protocol address — no proxy is needed.
///
/// @dev    Run via:
///           forge script contracts/script/RegisterBlueprint.s.sol \
///             --rpc-url $RPC_URL --broadcast --slow
///
///         The companion `deploy/register-blueprint.sh` wraps this script
///         and exports the resulting addresses + blueprint id for downstream
///         operator-registration scripts.
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real
    // chains (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // USDC on Base Sepolia. Training jobs settle in this token by default —
    // pass PAYMENT_TOKEN via env to override on other networks.
    address constant DEFAULT_PAYMENT_TOKEN = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        // Reserved for future BSM wiring (e.g. payment-token-aware billing).
        // Surfaced here so the bash wrapper / config layer have a single source
        // of truth for the settlement token.
        address paymentToken = vm.envOr("PAYMENT_TOKEN", DEFAULT_PAYMENT_TOKEN);

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy DistributedTrainingBSM ───────────────────────────────────
        DistributedTrainingBSM bsm = new DistributedTrainingBSM(tangleAddr);

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_TRAINING_BSM=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_TRAINING_BLUEPRINT_ID=%s", vm.toString(blueprintId));
        console2.log("DEPLOY_TRAINING_PAYMENT_TOKEN=%s", vm.toString(paymentToken));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/training-blueprint";
        // Until canonical metadata JSON is pinned via IPFS, derive the digest
        // from the metadataUri so the value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Multi-operator training: dynamic membership (operators join/leave as
        // hardware comes online) and event-driven pricing (operators settle
        // per-job at completion via DistributedTrainingBSM.distributePayment).
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 2, // matches BSM.createTrainingJob: `minOperators >= 2`
            maxOperators: 0, // unbounded — capped per-job by the BSM
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // operators negotiate price per training job via RFQ
        });

        def.metadata = Types.BlueprintMetadata({
            name: "Distributed Training Blueprint",
            description: "Multi-operator distributed model training (SFT/DPO/GRPO/pretrain) with checkpoint sync + GPU-minute proportional payouts",
            author: "Tangle",
            category: "AI/Training",
            codeRepository: "https://github.com/tangle-network/training-blueprint",
            logo: "",
            website: "https://tangle.network",
            license: "MIT",
            profilingData: ""
        });

        def.jobs = _buildJobs();

        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "training-operator",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/training-operator",
                "./target/release/training-operator"
            ),
            testing: Types.TestingSource("distributed-training", "training-operator", "."),
            binaries: bins
        });

        def.supportedMemberships = new Types.MembershipModel[](1);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
    }

    function _buildJobs() internal pure returns (Types.JobDefinition[] memory jobs) {
        // The Rust operator owns the typed shapes; on-chain schemas are kept
        // empty to match the proven ai-agent-sandbox + llm-inference pattern.
        // A future PR can introduce hex-encoded schemas via tnt-core's
        // SchemaLib once that surface stabilizes across blueprints.
        jobs = new Types.JobDefinition[](5);

        // Job 0: create a training job (anyone — typically a model owner)
        //   inputs:  (string baseModel, string datasetUrl, string method,
        //             uint32 totalEpochs, uint32 minOperators,
        //             uint32 maxOperators, uint64 syncIntervalSteps)
        //   outputs: (uint64 jobId)
        jobs[0] = Types.JobDefinition({
            name: "create_training_job",
            description: "Create a multi-operator training job (base model + dataset + method)",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });

        // Job 1: operator joins a training job
        //   inputs:  (uint64 jobId)
        //   outputs: ()
        jobs[1] = Types.JobDefinition({
            name: "join_training",
            description: "Operator joins an active training job (requires onRegister capabilities)",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });

        // Job 2: operator submits a checkpoint hash after a sync round
        //   inputs:  (uint64 jobId, bytes32 checkpointHash, uint32 epoch)
        //   outputs: ()
        jobs[2] = Types.JobDefinition({
            name: "submit_checkpoint",
            description: "Submit checkpoint hash after a sync round; auto-completes the job at totalEpochs",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });

        // Job 3: operator leaves a training job gracefully
        //   inputs:  (uint64 jobId)
        //   outputs: ()
        jobs[3] = Types.JobDefinition({
            name: "leave_training",
            description: "Operator leaves a training job gracefully (records leftAtEpoch)",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });

        // Job 4: distribute payment proportionally by GPU-minutes
        //   inputs:  (uint64 jobId)
        //   outputs: ()
        jobs[4] = Types.JobDefinition({
            name: "distribute_payment",
            description: "After job completes, distribute escrowed payment by GPU-minute contribution",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}
