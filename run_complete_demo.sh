#!/bin/bash
# Complete Demo: Baseline vs Checkpoint Restore Performance
# Compares cold start vs CRIU checkpoint restore for vLLM with Qwen 32B

set -e

PROMPT="${1:-What is the capital of France?}"

echo "========================================"
echo "GPU CHECKPOINT/RESTORE COMPLETE DEMO"
echo "========================================"
echo "Model: Qwen/Qwen2.5-32B-Instruct"
echo "Prompt: ${PROMPT}"
echo ""

# 1. Clean everything first
echo "=== Step 1: Cleaning up existing containers ==="
podman rm -f vllm-baseline-demo vllm-checkpoint 2>/dev/null || true
echo "Cleanup complete."
echo ""

# 2. Run baseline (cold start)
echo "=== Step 2: Running Baseline Demo (Cold Start) ==="
echo "This will take approximately 2-3 minutes..."
echo ""
/root/demo_baseline_final.sh "${PROMPT}"
echo ""

# Save baseline time
BASELINE_TIME=$(grep "Time to first inference:" /tmp/baseline_response.json 2>/dev/null || echo "N/A")

# 3. Clean up baseline
echo "=== Step 3: Cleaning up baseline container ==="
podman rm -f vllm-baseline-demo
echo "Baseline container removed."
echo ""

# 4. Create checkpoint
echo "=== Step 4: Creating Checkpoint ==="
echo "This will take approximately 2 minutes..."
echo ""
/root/create_checkpoint.sh
echo ""

# 5. Run checkpoint restore
echo "=== Step 5: Running Checkpoint Restore Demo ==="
echo "This will take approximately 30-35 seconds..."
echo ""
/root/demo_checkpoint_final.sh "${PROMPT}"
echo ""

# 6. Final cleanup
echo "=== Step 6: Final cleanup ==="
podman stop vllm-checkpoint 2>/dev/null || true
podman rm -f vllm-checkpoint
echo "Checkpoint container removed."
echo ""
