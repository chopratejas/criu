#!/bin/bash
# Checkpoint/Restore: Restore vLLM from checkpoint and measure time to first inference

set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
PROMPT="${1:-What is the capital of France?}"
CONTAINER_NAME="vllm-checkpoint"

echo "======================================"
echo "CHECKPOINT: Restore vLLM from Checkpoint"
echo "======================================"
echo ""

# Check if checkpoint exists
if ! podman container exists ${CONTAINER_NAME} 2>/dev/null; then
    echo "ERROR: No checkpoint found for container '${CONTAINER_NAME}'"
    echo "Please run ./create_checkpoint.sh first to create a checkpoint"
    exit 1
fi

# Make sure container is stopped
echo "Ensuring container is stopped..."
podman stop ${CONTAINER_NAME} 2>/dev/null || true

# Start timing
echo "Starting timer..."
START_TIME=$(date +%s.%N)

# Restore from checkpoint
echo "[$(date +%T)] Restoring container from checkpoint..."
RESTORE_START=$(date +%s.%N)
podman container restore ${CONTAINER_NAME}
RESTORE_END=$(date +%s.%N)
RESTORE_TIME=$(echo "$RESTORE_END - $RESTORE_START" | bc)

echo "[$(date +%T)] Container restored in ${RESTORE_TIME}s, waiting for vLLM to be ready..."

# Wait for vLLM to be ready
MAX_WAIT=60  # Should be much faster with checkpoint
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if port is responding
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "[$(date +%T)] vLLM is ready!"
        break
    fi
    sleep 0.5
    ELAPSED=$(echo "$ELAPSED + 0.5" | bc)
    if [ $(echo "$ELAPSED % 5 == 0" | bc) -eq 1 ]; then
        echo "[$(date +%T)] Still waiting... (${ELAPSED}s elapsed)"
    fi
done

if [ $(echo "$ELAPSED >= $MAX_WAIT" | bc) -eq 1 ]; then
    echo "ERROR: vLLM did not become ready within ${MAX_WAIT} seconds"
    podman logs ${CONTAINER_NAME} 2>&1 | tail -50
    exit 1
fi

# Send inference request
echo "[$(date +%T)] Sending inference request: '${PROMPT}'"
RESPONSE=$(curl -s http://localhost:8000/v1/completions \
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
echo "CHECKPOINT RESULTS"
echo "======================================"
echo "Model: ${MODEL}"
echo "Prompt: ${PROMPT}"
echo "Response: ${RESPONSE_TEXT}"
echo ""
echo "⚡ RESTORE TIME: ${RESTORE_TIME} seconds"
echo "⏱️  TOTAL TIME: ${TOTAL_TIME} seconds"
echo "======================================"
echo ""

# Stop container (don't remove, keep checkpoint)
echo "Stopping container (checkpoint preserved)..."
podman stop ${CONTAINER_NAME} >/dev/null 2>&1

echo "Checkpoint test complete!"
