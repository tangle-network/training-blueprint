// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import { Test } from "forge-std/Test.sol";
import "../src/DistributedTrainingBSM.sol";

contract DistributedTrainingBSMTest is Test {
    DistributedTrainingBSM bsm;
    address operatorA = address(2);
    address operatorB = address(3);
    address unregistered = address(4);

    function setUp() public {
        bsm = new DistributedTrainingBSM(address(this));
        // Configure a model tier
        // minVramMib=24000, minGpuCount=1, minBandwidthMbps=100
        bsm.configureModelTier("llama-3.1-8b", 24000, 1, 100);
    }

    function test_registerOperator() public {
        _register(operatorA);
        (,,,,, bool active) = bsm.operatorCaps(operatorA);
        assertTrue(active);
    }

    function test_createTrainingJob() public {
        _register(operatorA);
        uint64 jobId = bsm.createTrainingJob("llama-3.1-8b", "https://data.example.com/train.jsonl", "sft", 10, 2, 8, 500);
        assertGt(jobId, 0);
    }

    function test_joinTraining() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = bsm.createTrainingJob("llama-3.1-8b", "https://data.example.com", "sft", 10, 2, 8, 500);

        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        address[] memory ops = bsm.getJobOperators(jobId);
        assertEq(ops.length, 2);
        assertTrue(bsm.isOperatorInJob(jobId, operatorA));
        assertTrue(bsm.isOperatorInJob(jobId, operatorB));
    }

    function test_submitCheckpoint() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = _createJob();
        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        bytes32 hash = keccak256("epoch-1-checkpoint");
        vm.prank(operatorA);
        bsm.submitCheckpoint(jobId, hash, 1);
    }

    function test_leaveTraining() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = _createJob();
        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        vm.prank(operatorA);
        bsm.leaveTraining(jobId);

        assertFalse(bsm.isOperatorInJob(jobId, operatorA));
    }

    function test_updateContribution() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = _createJob();
        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        bsm.updateContribution(jobId, operatorA, 10, 100);
        (uint64 gpuMinutes, uint64 steps,,,) = bsm.contributions(jobId, operatorA);
        assertEq(gpuMinutes, 10);
        assertEq(steps, 100);
    }

    function test_cannotJoinUnregistered() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = _createJob();
        vm.expectRevert();
        vm.prank(unregistered);
        bsm.joinTraining(jobId);
    }

    function _register(address op) internal {
        // Register with enough VRAM (24GB meets the 24000 MiB model tier requirement)
        bsm.onRegister(op, abi.encode(uint32(1), uint32(48000), uint64(1000), "A100", "http://op"));
    }

    function _createJob() internal returns (uint64) {
        return bsm.createTrainingJob("llama-3.1-8b", "https://data.example.com", "sft", 10, 2, 8, 500);
    }
}
