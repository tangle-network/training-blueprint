#!/usr/bin/env bash
# Register the distributed-training blueprint on Tangle.
#
# Single-shot flow: deploys DistributedTrainingBSM AND calls
# Tangle.createBlueprint in the same broadcast via
# `contracts/script/RegisterBlueprint.s.sol`.
#
# DistributedTrainingBSM is a regular (non-upgradeable) contract — its
# constructor takes the Tangle protocol address, so no proxy is needed.
#
# Prerequisites:
#   - forge installed (used to deploy the BSM)
#   - cast installed (used to render chain info + encode registration calldata)
#   - cargo + the `cargo-tangle` CLI installed for the manifest step:
#       cargo install cargo-tangle --locked
#   - Deployer wallet funded on the target network
#
# Usage (Base Sepolia, against the already-deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export PAYMENT_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia
#   ./deploy/register-blueprint.sh
#
# Local anvil (LocalTestnet snapshot):
#
#   export RPC_URL=http://127.0.0.1:8545
#   ./deploy/register-blueprint.sh   # anvil deployer key + Tangle/USDC defaults
#
# Optional overrides for the per-operator registration calldata (emitted at
# the end of the run so an operator can self-register against the new id):
#   GPU_COUNT          GPUs the operator exposes (default: 1)
#   TOTAL_VRAM         Total VRAM in MiB (default: 48000)
#   BANDWIDTH_MBPS     Network bandwidth in Mbps (default: 1000)
#   GPU_MODEL          GPU model string (default: NVIDIA A100)
#   ENDPOINT           Operator HTTP endpoint (default: https://your-operator.example.com)
#
# Manifest registration (cargo-tangle) — the on-chain createBlueprint above
# only registers the BSM + Blueprint id. The Tangle runtime ALSO expects a
# manifest (`blueprint.json`) describing the operator binary, jobs, and
# registration schema; that is published via the cargo-tangle CLI. The CLI
# does NOT deploy the BSM (forge does, as above).
#
#   cargo tangle blueprint deploy tangle \
#       --network testnet \
#       --definition <path-to-blueprint.json> \
#       --tangle-contract "$TANGLE_CORE" \
#       --http-rpc-url "$RPC_URL" \
#       --ws-rpc-url   "${WS_RPC_URL:-${RPC_URL/https:/wss:}}" \
#       --keystore-path "${KEYSTORE_PATH:?Set KEYSTORE_PATH for cargo-tangle}"
#
# Outputs (parsed by deployment scripts, do not change without coordinating):
#   DEPLOY_TRAINING_BSM=<address>
#   DEPLOY_TRAINING_BLUEPRINT_ID=<u64>
#   DEPLOY_TRAINING_PAYMENT_TOKEN=<address>

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"

GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_VRAM="${TOTAL_VRAM:-48000}"
BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-1000}"
GPU_MODEL="${GPU_MODEL:-NVIDIA A100}"
ENDPOINT="${ENDPOINT:-https://your-operator.example.com}"

echo "=== Distributed Training Blueprint Registration ==="
echo "Network:        $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:       $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core:    ${TANGLE_CORE:-<default from RegisterBlueprint.s.sol>}"
echo "Payment Token:  ${PAYMENT_TOKEN:-<default USDC sepolia>}"
echo "Operator HW:    ${GPU_COUNT} x ${GPU_MODEL} (${TOTAL_VRAM} MiB, ${BANDWIDTH_MBPS} Mbps)"
echo "Endpoint:       ${ENDPOINT}"
echo ""

cd "$(dirname "$0")/../contracts"

# Deploy BSM AND register the blueprint in one forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    PAYMENT_TOKEN="${PAYMENT_TOKEN:-}" \
    forge script script/RegisterBlueprint.s.sol \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract addresses + blueprint id for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_TRAINING_BSM=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_TRAINING_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output"
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID:               $BLUEPRINT_ID"
echo "DistributedTrainingBSM:     $BSM_ADDRESS"
echo ""

# Per-operator registration is a separate step. Encode the registration
# calldata that DistributedTrainingBSM.onRegister expects:
#   (uint32 gpuCount, uint32 totalVramMib, uint64 networkBandwidthMbps,
#    string gpuModel, string endpoint)
REG_INPUTS=$(cast abi-encode \
    "f(uint32,uint32,uint64,string,string)" \
    "$GPU_COUNT" "$TOTAL_VRAM" "$BANDWIDTH_MBPS" "$GPU_MODEL" "$ENDPOINT")

echo "Operator registration inputs (use these to register an operator):"
echo "  $REG_INPUTS"
echo ""
echo "To register an operator now:"
echo "  cast send ${TANGLE_CORE:-<TANGLE_CORE>} \\"
echo "    'registerOperator(uint64,bytes)' $BLUEPRINT_ID $REG_INPUTS \\"
echo "    --rpc-url $RPC_URL --private-key \$OPERATOR_KEY"
