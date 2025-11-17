#!/bin/bash
# Create a checkpoint of vLLM with Qwen 32B after full initialization

set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
CONTAINER_NAME="vllm-checkpoint"

echo "======================================"
echo "Creating vLLM Checkpoint"
echo "======================================"
echo ""

# Clean up any existing container/checkpoint
echo "Cleaning up existing containers..."
podman rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Start vLLM container
echo "[$(date +%T)] Starting vLLM container with ${MODEL}..."
podman run -d \
  --name ${CONTAINER_NAME} \
  --device nvidia.com/gpu=all \
  --security-opt=seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --security-opt=label=disable \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:v0.11.0 \
  --model ${MODEL} \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --dtype auto

echo "[$(date +%T)] Waiting for model to load..."

# Wait for vLLM to be ready
MAX_WAIT=600  # 10 minutes
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if podman logs ${CONTAINER_NAME} 2>&1 | grep -q "Application startup complete"; then
        echo "[$(date +%T)] vLLM is ready!"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo "[$(date +%T)] Still loading... (${ELAPSED}s elapsed)"
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM did not become ready within ${MAX_WAIT} seconds"
    podman logs ${CONTAINER_NAME} 2>&1 | tail -50
    exit 1
fi

# Send warmup inference to trigger CUDA graph compilation
echo "[$(date +%T)] Sending warmup inference to compile CUDA graphs..."
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"Hello, world!\",
    \"max_tokens\": 10,
    \"temperature\": 0.7
  }" > /dev/null

echo "[$(date +%T)] Warmup complete, waiting 5 seconds for stabilization..."
sleep 5

# Create checkpoint
echo "[$(date +%T)] Creating checkpoint..."
CHECKPOINT_START=$(date +%s.%N)
podman container checkpoint ${CONTAINER_NAME}
CHECKPOINT_END=$(date +%s.%N)
CHECKPOINT_TIME=$(echo "$CHECKPOINT_END - $CHECKPOINT_START" | bc)

echo ""
echo "======================================"
echo "Checkpoint created successfully!"
echo "======================================"
echo "Container: ${CONTAINER_NAME}"
echo "Model: ${MODEL}"
echo "Checkpoint time: ${CHECKPOINT_TIME} seconds"
echo ""
echo "You can now run ./checkpoint.sh to test restore performance"
echo "======================================"
