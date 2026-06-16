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

        bsm.updateContribution(jobId, operatorA, 10, 100, true, 500);
        (uint64 gpuMinutes, uint64 steps,,,, bool certified, int64 improvementBps) =
            bsm.contributions(jobId, operatorA);
        assertEq(gpuMinutes, 10);
        assertEq(steps, 100);
        assertTrue(certified);
        assertEq(improvementBps, 500);
    }

    function test_cannotJoinUnregistered() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = _createJob();
        vm.expectRevert();
        vm.prank(unregistered);
        bsm.joinTraining(jobId);
    }

    /// An operator whose held-out certification is FALSE receives ZERO from
    /// distributePayment, while a certified operator is paid the full pot. This is the
    /// on-chain enforcement of "no pay for a non-improving checkpoint".
    function test_uncertifiedOperatorGetsZero() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = bsm.createTrainingJob{ value: 10 ether }(
            "llama-3.1-8b", "https://data.example.com", "sft", 1, 2, 8, 500
        );

        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        // Both operators did equal GPU-minutes of work...
        // operatorA's checkpoint cleared the held-out gate; operatorB's did NOT.
        bsm.updateContribution(jobId, operatorA, 100, 1000, true, 800);
        bsm.updateContribution(jobId, operatorB, 100, 1000, false, -300);

        // Complete the job (epoch >= totalEpochs auto-completes).
        bytes32 hash = keccak256("final");
        vm.prank(operatorA);
        bsm.submitCheckpoint(jobId, hash, 1);

        uint256 balABefore = operatorA.balance;
        uint256 balBBefore = operatorB.balance;

        bsm.distributePayment(jobId);

        // Certified operatorA takes the entire pot; uncertified operatorB gets ZERO.
        assertEq(operatorA.balance - balABefore, 10 ether);
        assertEq(operatorB.balance - balBBefore, 0);
    }

    /// Two certified operators with equal GPU-minutes split the pot 50/50 — the
    /// existing pro-rata accounting is preserved for certified contributions.
    function test_certifiedOperatorsSplitProRata() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = bsm.createTrainingJob{ value: 10 ether }(
            "llama-3.1-8b", "https://data.example.com", "sft", 1, 2, 8, 500
        );

        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        bsm.updateContribution(jobId, operatorA, 100, 1000, true, 800);
        bsm.updateContribution(jobId, operatorB, 100, 1000, true, 600);

        bytes32 hash = keccak256("final");
        vm.prank(operatorA);
        bsm.submitCheckpoint(jobId, hash, 1);

        uint256 balABefore = operatorA.balance;
        uint256 balBBefore = operatorB.balance;

        bsm.distributePayment(jobId);

        assertEq(operatorA.balance - balABefore, 5 ether);
        assertEq(operatorB.balance - balBBefore, 5 ether);
    }

    /// If NO operator is certified, there is nothing to distribute and the call reverts
    /// (fail-closed): the pot is not silently drained or paid to anyone.
    function test_noCertifiedContributionsReverts() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = bsm.createTrainingJob{ value: 10 ether }(
            "llama-3.1-8b", "https://data.example.com", "sft", 1, 2, 8, 500
        );

        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        bsm.updateContribution(jobId, operatorA, 100, 1000, false, -100);
        bsm.updateContribution(jobId, operatorB, 100, 1000, false, -200);

        bytes32 hash = keccak256("final");
        vm.prank(operatorA);
        bsm.submitCheckpoint(jobId, hash, 1);

        vm.expectRevert(bytes("no certified contributions"));
        bsm.distributePayment(jobId);
    }

    /// An operator cannot certify itself: updateContribution/recordCertification are
    /// gated to Tangle or the local admin, not arbitrary callers.
    function test_operatorCannotSelfCertify() public {
        _register(operatorA);
        _register(operatorB);
        uint64 jobId = _createJob();
        vm.prank(operatorA);
        bsm.joinTraining(jobId);
        vm.prank(operatorB);
        bsm.joinTraining(jobId);

        vm.expectRevert(bytes("only tangle or owner"));
        vm.prank(operatorA);
        bsm.recordCertification(jobId, operatorA, true, 999);

        vm.expectRevert(bytes("only tangle or owner"));
        vm.prank(operatorA);
        bsm.updateContribution(jobId, operatorA, 100, 1000, true, 999);
    }

    function _register(address op) internal {
        // Register with enough VRAM (24GB meets the 24000 MiB model tier requirement)
        bsm.onRegister(op, abi.encode(uint32(1), uint32(48000), uint64(1000), "A100", "http://op"));
    }

    function _createJob() internal returns (uint64) {
        return bsm.createTrainingJob("llama-3.1-8b", "https://data.example.com", "sft", 10, 2, 8, 500);
    }
}
