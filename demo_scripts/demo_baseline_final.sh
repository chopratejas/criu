#!/bin/bash
# Professional Demo: Baseline cold start performance

set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
PROMPT="${1:-What is the capital of France?}"
CONTAINER_NAME="vllm-baseline-demo"

echo "======================================"
echo "BASELINE COLD START DEMO"
echo "======================================"
echo "Model: ${MODEL}"
echo "Prompt: ${PROMPT}"
echo ""

# Clean up
podman rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Start timing
START_TIME=$(date +%s.%N)

# Start vLLM container
echo "Starting vLLM container..."
podman run -d \
  --name ${CONTAINER_NAME} \
  --device nvidia.com/gpu=all \
  --security-opt=seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --security-opt=label=disable \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8001:8000 \
  docker.io/vllm/vllm-openai:v0.11.0 \
  --model ${MODEL} \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --dtype auto >/dev/null

echo "Waiting for model to load..."

# Wait for vLLM
MAX_WAIT=600
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if podman logs ${CONTAINER_NAME} 2>&1 | grep -q "Application startup complete"; then
        echo "vLLM is ready!"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "Still loading... (${ELAPSED}s elapsed)"
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM did not start within ${MAX_WAIT} seconds"
    exit 1
fi

# Send inference request
echo "Sending inference request..."
RESPONSE=$(curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"${PROMPT}\",
    \"max_tokens\": 50,
    \"temperature\": 0.7
  }")

# Stop timing
END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# Extract response text
RESPONSE_TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].text' 2>/dev/null || echo "ERROR parsing response")

echo ""
echo "======================================"
echo "RESULTS"
echo "======================================"
echo "Response: ${RESPONSE_TEXT}"
echo ""
echo "Time to first inference: ${TOTAL_TIME} seconds"
echo "======================================"
echo ""

# Clean up
podman stop ${CONTAINER_NAME} >/dev/null 2>&1
podman rm ${CONTAINER_NAME} >/dev/null 2>&1

echo "Demo complete."
