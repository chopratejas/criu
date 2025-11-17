#!/bin/bash
# Professional Demo: Checkpoint restore performance

set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
PROMPT="${1:-What is the capital of France?}"
CONTAINER_NAME="vllm-checkpoint"

echo "======================================"
echo "CHECKPOINT RESTORE DEMO"
echo "======================================"
echo "Model: ${MODEL}"
echo "Prompt: ${PROMPT}"
echo ""

# Stop container if running
podman stop ${CONTAINER_NAME} 2>/dev/null || true

# Start timing
START_TIME=$(date +%s.%N)

# Restore from checkpoint
echo "Restoring container from checkpoint..."
RESTORE_START=$(date +%s.%N)
podman container restore --keep ${CONTAINER_NAME}
RESTORE_END=$(date +%s.%N)
RESTORE_TIME=$(echo "$RESTORE_END - $RESTORE_START" | bc)

echo "Container restored in ${RESTORE_TIME}s"
echo "Waiting for vLLM..."
sleep 2

# Send inference request
echo "Sending inference request..."
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

# Extract response text
RESPONSE_TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].text' 2>/dev/null || echo "ERROR parsing response")

echo ""
echo "======================================"
echo "RESULTS"
echo "======================================"
echo "Response: ${RESPONSE_TEXT}"
echo ""
echo "CRIU restore time: ${RESTORE_TIME} seconds"
echo "Time to first inference: ${TOTAL_TIME} seconds"
echo "======================================"
echo ""

# Stop container
podman stop ${CONTAINER_NAME} >/dev/null 2>&1

echo "Demo complete."
