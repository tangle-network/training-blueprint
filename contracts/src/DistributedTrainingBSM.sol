// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

/// @title DistributedTrainingBSM
/// @notice Blueprint Service Manager for multi-operator distributed model training.
/// Manages training jobs, operator lifecycle, checkpoint submission, and payment distribution.
/// @dev Inherits {BlueprintServiceManagerBase} so Tangle's `onBlueprintCreated`,
///      `onRegister`, `onJobCall`, etc. dispatch to safe defaults; the training-specific
///      surface only overrides `onRegister` to capture operator hardware capabilities.
contract DistributedTrainingBSM is BlueprintServiceManagerBase {
    // --- Structs ---

    struct TrainingJob {
        string baseModel;
        string datasetUrl;
        string method; // "sft", "dpo", "grpo", "pretrain"
        uint32 totalEpochs;
        uint32 currentEpoch;
        uint32 minOperators;
        uint32 maxOperators;
        uint64 syncIntervalSteps;
        address[] operators;
        bytes32 latestCheckpointHash;
        bool completed;
        uint256 totalPayment;
        address creator;
    }

    struct OperatorContribution {
        uint64 gpuMinutesContributed;
        uint64 stepsCompleted;
        uint32 joinedAtEpoch;
        uint32 leftAtEpoch;
        bool slashed;
        /// @notice Whether this operator's final checkpoint cleared the held-out
        ///         evaluation gate (bootstrap CI lower bound above the protocol
        ///         margin), decoded off-chain from the submitted `TrainingJobResult`
        ///         and recorded through the authenticated result path. {distributePayment}
        ///         pays ZERO to an operator whose contribution is not certified.
        bool heldOutCertified;
        /// @notice Certified held-out improvement in signed basis points (1e4 scale);
        ///         negative is a regression. Advisory/audit metadata — payout gates on
        ///         the {heldOutCertified} bool, which the off-chain gate sets only when
        ///         the CI lower bound cleared the margin.
        int64 improvementBps;
    }

    struct OperatorCapabilities {
        uint32 gpuCount;
        uint32 totalVramMib;
        uint64 networkBandwidthMbps;
        string gpuModel;
        string endpoint;
        bool active;
    }

    struct ModelTier {
        uint32 minVramMib;
        uint32 minGpuCount;
        uint64 minBandwidthMbps;
    }

    // --- State ---

    /// @notice Local admin (deployer). Separate from {BlueprintServiceManagerBase.blueprintOwner},
    ///         which is bound by Tangle when `onBlueprintCreated` fires.
    address public owner;

    uint64 public nextJobId;
    mapping(uint64 => TrainingJob) public jobs;
    mapping(uint64 => mapping(address => OperatorContribution)) public contributions;
    mapping(address => OperatorCapabilities) public operatorCaps;
    mapping(string => ModelTier) public modelTiers;

    // --- Events ---

    event JobCreated(uint64 indexed jobId, string baseModel, string method, address creator);
    event OperatorJoined(uint64 indexed jobId, address indexed operator, uint32 epoch);
    event OperatorLeft(uint64 indexed jobId, address indexed operator, uint32 epoch);
    event CheckpointSubmitted(uint64 indexed jobId, bytes32 checkpointHash, uint32 epoch);
    event JobCompleted(uint64 indexed jobId, bytes32 finalCheckpoint);
    event PaymentDistributed(uint64 indexed jobId, address indexed operator, uint256 amount);
    event OperatorRegistered(address indexed operator, uint32 gpuCount, uint32 vramMib);
    event OperatorSlashed(uint64 indexed jobId, address indexed operator, string reason);
    event ContributionCertified(uint64 indexed jobId, address indexed operator, bool certified, int64 improvementBps);

    // --- Modifiers ---

    modifier onlyOwner() {
        require(msg.sender == owner, "only owner");
        _;
    }

    /// @notice Allows either Tangle or the local admin (owner). Used by post-registration
    ///         maintenance entry points such as {slashOperator} and {updateContribution}.
    /// @dev    {BlueprintServiceManagerBase.onlyFromTangle} is strictly tangle-only; this
    ///         modifier preserves the previous "tangle OR owner" surface for ops tooling.
    modifier onlyTangleOrOwner() {
        require(msg.sender == tangleCore || msg.sender == owner, "only tangle or owner");
        _;
    }

    // --- Constructor ---

    /// @param _tangleCore Tangle protocol address. If non-zero, this binds the BSM to that
    ///        Tangle deployment up-front (so Tangle's own `onBlueprintCreated` call during
    ///        `createBlueprint` is a no-op rebind to the same address; see
    ///        {onBlueprintCreated} below). Passing `address(0)` defers binding to Tangle.
    constructor(address _tangleCore) {
        owner = msg.sender;
        tangleCore = _tangleCore;
        nextJobId = 1;
    }

    // --- Blueprint Lifecycle ---

    /// @notice Tangle dispatches this from `createBlueprint`. The base implementation reverts
    ///         with `AlreadyInitialized` if `tangleCore` is already set, which would happen
    ///         here because the constructor pre-binds it. Override to be idempotent: accept
    ///         a rebind to the same Tangle, record `blueprintId` + `blueprintOwner`.
    /// @dev    Without this override, Tangle's hook call reverts and `createBlueprint` aborts
    ///         with `ManagerRejected(manager)` (selector 0x71cc0e9a).
    function onBlueprintCreated(uint64 _blueprintId, address _owner, address _tangleCore) external override {
        // First-bind path: constructor was given address(0), let Tangle bind itself.
        if (tangleCore == address(0)) {
            tangleCore = _tangleCore;
        } else {
            // Pre-bound path: only accept the rebind if Tangle matches what the deployer set.
            require(_tangleCore == tangleCore, "tangle mismatch");
        }

        blueprintId = _blueprintId;
        blueprintOwner = _owner;
    }

    // --- Operator Registration ---

    /// @notice Capture operator hardware capabilities. Tangle invokes this when an operator
    ///         registers against the blueprint.
    /// @dev    Overrides {BlueprintServiceManagerBase.onRegister}; must remain `external payable`
    ///         and use the base `onlyFromTangle` modifier (tangle-only) to keep the
    ///         {IBlueprintServiceManager} ABI intact.
    function onRegister(address operator, bytes calldata registrationData) external payable override onlyFromTangle {
        (
            uint32 gpuCount,
            uint32 totalVramMib,
            uint64 networkBandwidthMbps,
            string memory gpuModel,
            string memory endpoint
        ) = abi.decode(registrationData, (uint32, uint32, uint64, string, string));

        operatorCaps[operator] = OperatorCapabilities({
            gpuCount: gpuCount,
            totalVramMib: totalVramMib,
            networkBandwidthMbps: networkBandwidthMbps,
            gpuModel: gpuModel,
            endpoint: endpoint,
            active: true
        });

        emit OperatorRegistered(operator, gpuCount, totalVramMib);
    }

    // --- Job Lifecycle ---

    /// @notice Create a new training job.
    function createTrainingJob(
        string calldata baseModel,
        string calldata datasetUrl,
        string calldata method,
        uint32 totalEpochs,
        uint32 minOperators,
        uint32 maxOperators,
        uint64 syncIntervalSteps
    )
        external
        payable
        returns (uint64 jobId)
    {
        require(totalEpochs > 0, "epochs must be > 0");
        require(minOperators >= 2, "min 2 operators");
        require(maxOperators >= minOperators, "max >= min operators");

        jobId = nextJobId++;

        TrainingJob storage job = jobs[jobId];
        job.baseModel = baseModel;
        job.datasetUrl = datasetUrl;
        job.method = method;
        job.totalEpochs = totalEpochs;
        job.currentEpoch = 0;
        job.minOperators = minOperators;
        job.maxOperators = maxOperators;
        job.syncIntervalSteps = syncIntervalSteps;
        job.totalPayment = msg.value;
        job.creator = msg.sender;

        emit JobCreated(jobId, baseModel, method, msg.sender);
    }

    /// @notice Operator joins an active training job.
    function joinTraining(uint64 jobId) external {
        TrainingJob storage job = jobs[jobId];
        require(!job.completed, "job completed");
        require(operatorCaps[msg.sender].active, "not registered");
        require(job.operators.length < job.maxOperators, "job full");
        require(!_isOperatorInJob(jobId, msg.sender), "already joined");
        require(_canJoin(jobId, msg.sender), "insufficient hardware");

        job.operators.push(msg.sender);
        contributions[jobId][msg.sender].joinedAtEpoch = job.currentEpoch;

        emit OperatorJoined(jobId, msg.sender, job.currentEpoch);
    }

    /// @notice Operator leaves a training job gracefully.
    function leaveTraining(uint64 jobId) external {
        require(_isOperatorInJob(jobId, msg.sender), "not in job");

        TrainingJob storage job = jobs[jobId];
        contributions[jobId][msg.sender].leftAtEpoch = job.currentEpoch;

        // Remove from active operators
        _removeOperator(jobId, msg.sender);

        emit OperatorLeft(jobId, msg.sender, job.currentEpoch);
    }

    /// @notice Submit a checkpoint hash after a sync round.
    function submitCheckpoint(uint64 jobId, bytes32 checkpointHash, uint32 epoch) external {
        require(_isOperatorInJob(jobId, msg.sender), "not in job");

        TrainingJob storage job = jobs[jobId];
        job.latestCheckpointHash = checkpointHash;
        job.currentEpoch = epoch;

        emit CheckpointSubmitted(jobId, checkpointHash, epoch);

        // Auto-complete if all epochs done
        if (epoch >= job.totalEpochs) {
            job.completed = true;
            emit JobCompleted(jobId, checkpointHash);
        }
    }

    /// @notice Called by Tangle when an operator submits a job result.
    /// @dev Decodes the blueprint-specific `TrainingJobResult` and persists the
    ///      held-out certification verdict for the submitting operator. This is the
    ///      authenticated, operator-agnostic path: the operator cannot mark itself
    ///      certified because Tangle is the caller. GPU-minutes and step counts are
    ///      intentionally NOT written here; they are supplied separately by the
    ///      verified aggregator via {updateContribution} so the two trust domains
    ///      (result format vs. resource accounting) stay separate.
    function onJobResult(
        uint64,
        uint8,
        uint64,
        address operator,
        bytes calldata,
        bytes calldata outputs
    )
        external
        payable
        override
        onlyFromTangle
    {
        // TrainingJobResult ABI:
        // (uint64 jobId, bytes32 finalCheckpointHash, uint64 totalSteps,
        //  uint32 finalEpoch, bool heldOutCertified, int64 improvementBps,
        //  int64 ciLowerBoundBps, uint32 heldOutExamples)
        (uint64 jobId,,,, bool heldOutCertified, int64 improvementBps,,) =
            abi.decode(outputs, (uint64, bytes32, uint64, uint32, bool, int64, int64, uint32));

        // The result must correspond to a job this operator actually joined.
        require(_isOperatorInJob(jobId, operator), "operator not in job");

        // Record only the certification verdict from the decoded result. The off-chain
        // held-out gate sets `heldOutCertified` to true only when the bootstrap CI lower
        // bound cleared the protocol margin; {distributePayment} pays ZERO if this flag
        // is false.
        OperatorContribution storage c = contributions[jobId][operator];
        c.heldOutCertified = heldOutCertified;
        c.improvementBps = improvementBps;

        // Emit the same event as the explicit certification entry points so indexers
        // have a single event to watch.
        emit ContributionCertified(jobId, operator, heldOutCertified, improvementBps);

        // Accept the result. Returning (bool) is not part of this interface, but the
        // base implementation does not return a value either; the hook records state
        // and proceeds.
    }

    /// @notice Distribute payment proportionally by GPU-minutes after job completion.
    /// @dev An operator is paid only if its contribution cleared the held-out evaluation
    ///      gate ({OperatorContribution.heldOutCertified}), which the authenticated result
    ///      path ({onJobResult}/{updateContribution}/{recordCertification}) sets from the
    ///      decoded `TrainingJobResult`. A non-improving/regressing checkpoint is reported
    ///      uncertified off-chain, so its operator is excluded from BOTH the denominator
    ///      and the payout loop here and receives ZERO. Certified operators keep the
    ///      existing GPU-minutes/slash pro-rata accounting.
    function distributePayment(uint64 jobId) external {
        TrainingJob storage job = jobs[jobId];
        require(job.completed, "not completed");
        require(job.totalPayment > 0, "no payment");

        uint64 totalGpuMinutes = 0;
        uint256 operatorCount = job.operators.length;

        for (uint256 i = 0; i < operatorCount; i++) {
            address op = job.operators[i];
            if (_isPayable(jobId, op)) {
                totalGpuMinutes += contributions[jobId][op].gpuMinutesContributed;
            }
        }

        require(totalGpuMinutes > 0, "no certified contributions");

        uint256 remaining = job.totalPayment;
        job.totalPayment = 0;

        for (uint256 i = 0; i < operatorCount; i++) {
            address op = job.operators[i];
            if (_isPayable(jobId, op) && contributions[jobId][op].gpuMinutesContributed > 0) {
                uint256 share = (remaining * contributions[jobId][op].gpuMinutesContributed) / totalGpuMinutes;
                payable(op).transfer(share);
                emit PaymentDistributed(jobId, op, share);
            }
        }
    }

    /// @notice An operator is payable iff it is not slashed AND its contribution
    ///         cleared the held-out evaluation gate. Single chokepoint so the
    ///         certification requirement cannot be bypassed by either accrual loop.
    function _isPayable(uint64 jobId, address operator) internal view returns (bool) {
        OperatorContribution storage c = contributions[jobId][operator];
        return !c.slashed && c.heldOutCertified;
    }

    /// @notice Configure hardware requirements for a model tier.
    function configureModelTier(
        string calldata modelName,
        uint32 minVramMib,
        uint32 minGpuCount,
        uint64 minBandwidthMbps
    )
        external
        onlyOwner
    {
        modelTiers[modelName] =
            ModelTier({ minVramMib: minVramMib, minGpuCount: minGpuCount, minBandwidthMbps: minBandwidthMbps });
    }

    /// @notice Slash an operator for misbehavior (called by verification).
    function slashOperator(uint64 jobId, address operator, string calldata reason) external onlyTangleOrOwner {
        require(_isOperatorInJob(jobId, operator), "not in job");
        contributions[jobId][operator].slashed = true;
        emit OperatorSlashed(jobId, operator, reason);
    }

    /// @notice Update operator contribution metrics (GPU-minutes and steps) without
    ///         touching the held-out certification. The authenticated certification
    ///         path ({onJobResult}/{recordCertification}) owns `heldOutCertified`
    ///         separately so metrics can be updated without accidentally overwriting
    ///         a certification verdict decoded from a `TrainingJobResult`.
    /// @dev    Authenticated by {onlyTangleOrOwner}: only Tangle (the legitimate result
    ///         submitter) or the local admin may set these. There is deliberately NO
    ///         path for an operator to update its own contribution.
    function updateContribution(
        uint64 jobId,
        address operator,
        uint64 gpuMinutes,
        uint64 steps
    )
        external
        onlyTangleOrOwner
    {
        OperatorContribution storage c = contributions[jobId][operator];
        c.gpuMinutesContributed = gpuMinutes;
        c.stepsCompleted = steps;
    }

    /// @notice Record only the held-out certification for an operator, without
    ///         touching GPU-minutes/steps. For flows that submit metrics and the
    ///         certificate separately.
    /// @dev    Same authentication as {updateContribution}: result-submitter only.
    function recordCertification(
        uint64 jobId,
        address operator,
        bool certified,
        int64 improvementBps
    )
        external
        onlyTangleOrOwner
    {
        OperatorContribution storage c = contributions[jobId][operator];
        c.heldOutCertified = certified;
        c.improvementBps = improvementBps;
        emit ContributionCertified(jobId, operator, certified, improvementBps);
    }

    // --- View Functions ---

    function getJobOperators(uint64 jobId) external view returns (address[] memory) {
        return jobs[jobId].operators;
    }

    function getJobStatus(uint64 jobId)
        external
        view
        returns (
            uint32 currentEpoch,
            uint32 totalEpochs,
            uint256 operatorCount,
            bool completed,
            bytes32 latestCheckpoint
        )
    {
        TrainingJob storage job = jobs[jobId];
        return (job.currentEpoch, job.totalEpochs, job.operators.length, job.completed, job.latestCheckpointHash);
    }

    function isOperatorInJob(uint64 jobId, address operator) external view returns (bool) {
        return _isOperatorInJob(jobId, operator);
    }

    // --- Internal ---

    function _isOperatorInJob(uint64 jobId, address operator) internal view returns (bool) {
        address[] storage ops = jobs[jobId].operators;
        for (uint256 i = 0; i < ops.length; i++) {
            if (ops[i] == operator) return true;
        }
        return false;
    }

    function _canJoin(uint64 jobId, address operator) internal view returns (bool) {
        OperatorCapabilities storage caps = operatorCaps[operator];
        TrainingJob storage job = jobs[jobId];

        // Check model tier requirements if configured
        ModelTier storage tier = modelTiers[job.baseModel];
        if (tier.minVramMib > 0) {
            if (caps.totalVramMib < tier.minVramMib) return false;
            if (caps.gpuCount < tier.minGpuCount) return false;
            if (caps.networkBandwidthMbps < tier.minBandwidthMbps) return false;
        }

        return true;
    }

    function _removeOperator(uint64 jobId, address operator) internal {
        address[] storage ops = jobs[jobId].operators;
        for (uint256 i = 0; i < ops.length; i++) {
            if (ops[i] == operator) {
                ops[i] = ops[ops.length - 1];
                ops.pop();
                return;
            }
        }
    }
}
