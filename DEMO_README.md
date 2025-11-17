# vLLM Checkpoint/Restore Demo

This demo showcases the performance benefits of checkpoint/restore for GPU inference workloads using vLLM and CRIU.

## What This Demo Shows

**Problem**: Cold-starting large language models (like Qwen 32B) takes significant time:
- Loading model weights from disk (~60GB)
- Initializing CUDA contexts
- Compiling CUDA graphs for optimized inference
- All of this happens before the first inference request can be served

**Solution**: Checkpoint/Restore allows you to:
1. Pre-warm a container with the model fully loaded
2. Save the entire GPU + CPU state to disk
3. Restore it instantly when needed (seconds instead of minutes)

## Use Cases

- **Serverless Inference**: Reduce cold start latency dramatically
- **Spot Instances**: Quick recovery when instance is reclaimed
- **Scaling**: Rapid scale-up from 0→N replicas
- **Cost Optimization**: Pay only for active inference time, not loading time

## Scripts

### 1. `create_checkpoint.sh`
Creates a checkpoint of vLLM with Qwen 32B model after full initialization.
```bash
./create_checkpoint.sh
```

### 2. `baseline.sh`
Measures cold start time: container start → first inference response
```bash
./baseline.sh "What is the capital of France?"
```

### 3. `checkpoint.sh`
Measures restore time: checkpoint restore → first inference response
```bash
./checkpoint.sh "What is the capital of France?"
```

### 4. `compare_demo.sh` (Recommended)
Runs both tests and shows the speedup comparison
```bash
./compare_demo.sh "What is the capital of France?"
```

## Quick Start

Run the full comparison demo:
```bash
./compare_demo.sh
```

This will:
1. Create a checkpoint if one doesn't exist (takes ~5-10 minutes)
2. Run baseline cold start test
3. Run checkpoint restore test
4. Show comparison with speedup metrics

## Expected Results

**Baseline (Cold Start)**:
- Model loading: ~3-5 minutes
- CUDA graph compilation: ~30-60 seconds
- **Total**: ~4-6 minutes

**Checkpoint (Restore)**:
- GPU restore: ~10-15 seconds
- Container restore: ~30-35 seconds
- **Total**: ~35-50 seconds

**Speedup**: ~6-8x faster time-to-first-token

## Technical Details

- **Model**: Qwen/Qwen2.5-32B-Instruct (~60GB)
- **Framework**: vLLM v0.11.0
- **GPU**: NVIDIA H100 (80GB VRAM)
- **Checkpoint Tool**: CRIU with custom GPU checkpoint plugin
- **Container Runtime**: Podman with custom CRIU integration

## Architecture

```
┌─────────────────────────────────────────┐
│         vLLM Container                  │
│  ┌──────────────────────────────────┐   │
│  │  Python Process                  │   │
│  │  - Model weights in GPU memory   │   │
│  │  - CUDA graphs compiled          │   │
│  │  - HTTP server ready             │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
                 ↓ checkpoint
┌─────────────────────────────────────────┐
│     Checkpoint Files (~75GB)            │
│  - CPU memory pages                     │
│  - GPU VRAM state                       │
│  - File descriptors                     │
│  - Network connections                  │
└─────────────────────────────────────────┘
                 ↓ restore (30s)
┌─────────────────────────────────────────┐
│    Restored vLLM Container              │
│  - All state instantly restored         │
│  - Ready for inference immediately      │
└─────────────────────────────────────────┘
```

## Performance Optimizations

This demo uses a custom-optimized CRIU with:
- **256 parallel workers** for VMA (memory) restoration
- **Per-worker file descriptors** to eliminate kernel lock contention
- **MADV_HUGEPAGE** hints for transparent huge pages
- **Custom CUDA plugin** for GPU state checkpoint/restore

These optimizations reduce restore time from ~13 minutes (vanilla CRIU) to ~30 seconds (13x speedup).

## Cleanup

To remove all demo containers and checkpoints:
```bash
podman rm -f vllm-baseline-demo vllm-checkpoint-demo
```

## Notes

- First run will download Qwen 32B model (~60GB) to HuggingFace cache
- Checkpoint files are stored in podman's checkpoint directory
- Each checkpoint is ~75GB (GPU VRAM + CPU memory)
- GPU must have sufficient VRAM for the model (32B needs ~60GB)
