<p align="center">
  <a href="https://tangle.tools" target="_blank">
    <img src="https://raw.githubusercontent.com/tangle-network/tangle/main/assets/Tangle_Logo_Full.png" alt="Tangle Logo" width="400">
  </a>
</p>

<h1 align="center">Training Blueprint</h1>

<p align="center">
  <a href="https://github.com/tangle-network/training-blueprint/actions"><img src="https://img.shields.io/github/actions/workflow/status/tangle-network/training-blueprint/ci.yml?branch=main&style=flat-square" alt="Build Status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://discord.gg/tangle"><img src="https://img.shields.io/discord/tangle?style=flat-square" alt="Discord"></a>
</p>

<p align="center">
  Permissionless multi-operator model training on Tangle using DeMo (Decoupled Momentum Optimization) for 10,000x communication reduction.
</p>

---

## Overview

This Blueprint enables distributed model training across permissionless operators. Operators join and leave freely at epoch boundaries, the protocol coordinates work via the DeMo algorithm, and the Tangle chain enforces payment and slashing.

**DeMo** (from [Nous Research](https://arxiv.org/abs/2411.19870)) reduces inter-operator communication by 10,000x compared to naive gradient synchronization:

1. Each operator trains locally with AdamW on its data shard
2. Every K steps, operators synchronize by applying DCT to momentum buffers
3. Top-k sparsification keeps only 0.1% of DCT coefficients (~100KB vs ~10GB)
4. Compressed momentum is broadcast via libp2p gossip and aggregated
5. Inverse DCT reconstructs the shared momentum update

## Components

| Component | Description |
|---|---|
| `operator/src/demo.rs` | DeMo optimizer — DCT, top-k sparsification, aggregation |
| `operator/src/coordinator.rs` | Multi-operator coordination — shard assignment, join/leave, sync barriers |
| `operator/src/network.rs` | libp2p gossip networking for momentum sync and checkpoint transfer |
| `operator/src/training.rs` | Training backend interface (axolotl, unsloth, torchtune) |
| `operator/src/checkpoint.rs` | Checkpoint save/load/hash for on-chain submission |
| `operator/src/verification.rs` | TOPLOC-inspired state transition proofs and gradient norm verification |
| `operator/src/server.rs` | HTTP API for job management |
| `operator/src/qos.rs` | Heartbeat with training-specific metrics |
| `contracts/` | DistributedTrainingBSM Solidity contract |

## Supported Methods

| Method | Description |
|---|---|
| **SFT** | Supervised fine-tuning on instruction datasets |
| **DPO** | Direct Preference Optimization for alignment |
| **GRPO** | Group Relative Policy Optimization |
| **Pretrain** | Continued pre-training on domain corpora |

## TEE Support

Operators run inside Trusted Execution Environments (Intel TDX / AMD SEV). The `TeeLayer` in the job router ensures all on-chain job results are attested, preventing operators from submitting fabricated training outputs.

## Quick Start

```bash
# Build the operator
cargo build --release

# Configure (see operator config for all options)
export TRAIN_OP_TANGLE__RPC_URL=https://rpc.tangle.tools
export TRAIN_OP_TANGLE__OPERATOR_KEY=0x...
export TRAIN_OP_TRAINING__ENDPOINT=http://localhost:5000
export TRAIN_OP_GPU__GPU_COUNT=4
export TRAIN_OP_GPU__TOTAL_VRAM_MIB=327680

# Run
./target/release/training-operator
```

## Architecture

```
                    Tangle Chain
                         |
              DistributedTrainingBSM
              /     |     |      \
         Op A    Op B   Op C    Op D    (permissionless operators)
          |        |      |       |
        shard0  shard1  shard2  shard3  (data parallelism)
          |        |      |       |
        AdamW   AdamW   AdamW   AdamW   (local training)
          \       |      |       /
           \      |      |      /
            DeMo Sync (every K steps)
            - DCT transform momentum
            - Top-k sparsify (0.1%)
            - Gossip broadcast (~100KB)
            - Aggregate + inverse DCT
```

## Related Repositories

- [blueprint](https://github.com/tangle-network/blueprint) — Blueprint SDK
- [vllm-inference-blueprint](https://github.com/tangle-network/vllm-inference-blueprint) — Single-operator inference Blueprint
- [gadget](https://github.com/tangle-network/gadget) — Tangle operator node

## References

- [DeMo / DisTrO](https://arxiv.org/abs/2411.19870) — Nous Research, 10,000x communication reduction
- [OpenDiLoCo](https://arxiv.org/abs/2407.07852) — Prime Intellect training
- [INTELLECT-2](https://arxiv.org/abs/2505.07291) — Async GRPO with TOPLOC verification

## License

MIT OR Apache-2.0
