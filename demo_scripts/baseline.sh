#!/bin/bash
# Baseline: Cold start vLLM with Qwen 32B model and measure time to first inference

set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
PROMPT="${1:-What is the capital of France?}"
CONTAINER_NAME="vllm-baseline-demo"

echo "======================================"
echo "BASELINE: Cold Start vLLM with ${MODEL}"
echo "======================================"
echo ""

# Clean up any existing container
echo "Cleaning up existing containers..."
podman rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Start timing
echo "Starting timer..."
START_TIME=$(date +%s.%N)

# Start vLLM container
echo "[$(date +%T)] Starting vLLM container..."
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
  --dtype auto

echo "[$(date +%T)] Container started, waiting for model to load..."

# Wait for vLLM to be ready by checking logs
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
        echo "[$(date +%T)] Still waiting... (${ELAPSED}s elapsed)"
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM did not become ready within ${MAX_WAIT} seconds"
    podman logs ${CONTAINER_NAME} 2>&1 | tail -50
    exit 1
fi

# Send inference request
echo "[$(date +%T)] Sending inference request: '${PROMPT}'"
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

# Extract response text (handle both .text and .message.content formats)
if echo "$RESPONSE" | jq -e '.choices[0].text' >/dev/null 2>&1; then
    RESPONSE_TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].text')
elif echo "$RESPONSE" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    RESPONSE_TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content')
else
    RESPONSE_TEXT=$(echo "$RESPONSE" | head -c 200)
fi

echo ""
echo "======================================"
echo "BASELINE RESULTS"
echo "======================================"
echo "Model: ${MODEL}"
echo "Prompt: ${PROMPT}"
echo "Response: ${RESPONSE_TEXT}"
echo ""
echo "⏱️  TOTAL TIME: ${TOTAL_TIME} seconds"
echo "======================================"
echo ""

# Clean up
echo "Stopping container..."
podman stop ${CONTAINER_NAME} >/dev/null 2>&1
podman rm ${CONTAINER_NAME} >/dev/null 2>&1

echo "Baseline test complete!"
