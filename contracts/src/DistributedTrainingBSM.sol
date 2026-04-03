// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title DistributedTrainingBSM
/// @notice Blueprint Service Manager for multi-operator distributed model training.
/// Manages training jobs, operator lifecycle, checkpoint submission, and payment distribution.
contract DistributedTrainingBSM {
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
        uint64 gpuHoursContributed;
        uint64 stepsCompleted;
        uint32 joinedAtEpoch;
        uint32 leftAtEpoch;
        bool slashed;
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

    address public owner;
    address public tangleCore;

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

    // --- Modifiers ---

    modifier onlyOwner() {
        require(msg.sender == owner, "only owner");
        _;
    }

    modifier onlyFromTangle() {
        require(msg.sender == tangleCore || msg.sender == owner, "only from tangle");
        _;
    }

    // --- Constructor ---

    constructor(address _tangleCore) {
        owner = msg.sender;
        tangleCore = _tangleCore;
        nextJobId = 1;
    }

    // --- Operator Registration ---

    /// @notice Register operator capabilities. Called during Blueprint onRegister.
    function onRegister(
        address operator,
        bytes calldata registrationData
    ) external onlyFromTangle {
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
    ) external payable returns (uint64 jobId) {
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
    function submitCheckpoint(
        uint64 jobId,
        bytes32 checkpointHash,
        uint32 epoch
    ) external {
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

    /// @notice Distribute payment proportionally by GPU-hours after job completion.
    function distributePayment(uint64 jobId) external {
        TrainingJob storage job = jobs[jobId];
        require(job.completed, "not completed");
        require(job.totalPayment > 0, "no payment");

        uint64 totalGpuHours = 0;
        uint256 operatorCount = job.operators.length;

        for (uint256 i = 0; i < operatorCount; i++) {
            address op = job.operators[i];
            if (!contributions[jobId][op].slashed) {
                totalGpuHours += contributions[jobId][op].gpuHoursContributed;
            }
        }

        require(totalGpuHours > 0, "no contributions");

        uint256 remaining = job.totalPayment;
        job.totalPayment = 0;

        for (uint256 i = 0; i < operatorCount; i++) {
            address op = job.operators[i];
            if (!contributions[jobId][op].slashed && contributions[jobId][op].gpuHoursContributed > 0) {
                uint256 share = (remaining * contributions[jobId][op].gpuHoursContributed) / totalGpuHours;
                payable(op).transfer(share);
                emit PaymentDistributed(jobId, op, share);
            }
        }
    }

    /// @notice Configure hardware requirements for a model tier.
    function configureModelTier(
        string calldata modelName,
        uint32 minVramMib,
        uint32 minGpuCount,
        uint64 minBandwidthMbps
    ) external onlyOwner {
        modelTiers[modelName] = ModelTier({
            minVramMib: minVramMib,
            minGpuCount: minGpuCount,
            minBandwidthMbps: minBandwidthMbps
        });
    }

    /// @notice Slash an operator for misbehavior (called by verification).
    function slashOperator(
        uint64 jobId,
        address operator,
        string calldata reason
    ) external onlyFromTangle {
        require(_isOperatorInJob(jobId, operator), "not in job");
        contributions[jobId][operator].slashed = true;
        emit OperatorSlashed(jobId, operator, reason);
    }

    /// @notice Update operator contribution metrics (called by heartbeat processor).
    function updateContribution(
        uint64 jobId,
        address operator,
        uint64 gpuHours,
        uint64 steps
    ) external onlyFromTangle {
        contributions[jobId][operator].gpuHoursContributed = gpuHours;
        contributions[jobId][operator].stepsCompleted = steps;
    }

    // --- View Functions ---

    function getJobOperators(uint64 jobId) external view returns (address[] memory) {
        return jobs[jobId].operators;
    }

    function getJobStatus(uint64 jobId) external view returns (
        uint32 currentEpoch,
        uint32 totalEpochs,
        uint256 operatorCount,
        bool completed,
        bytes32 latestCheckpoint
    ) {
        TrainingJob storage job = jobs[jobId];
        return (
            job.currentEpoch,
            job.totalEpochs,
            job.operators.length,
            job.completed,
            job.latestCheckpointHash
        );
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
