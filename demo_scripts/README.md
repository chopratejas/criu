# GPU Checkpoint/Restore Demo Scripts

This directory contains demo scripts and documentation for GPU checkpoint/restore with CRIU + cuda-checkpoint for vLLM inference.

## Overview

These scripts demonstrate the performance benefits of using CRIU (Checkpoint/Restore in Userspace) with NVIDIA's cuda-checkpoint to enable fast startup of large language model inference servers.

**Key Results:**
- **Model**: Qwen 2.5 32B (~60GB weights, 86GB total memory)
- **Hardware**: 1x H100 GPU (80GB)
- **Baseline cold start**: ~160 seconds
- **Checkpoint restore**: ~33 seconds
- **Speedup**: 4.8x faster with checkpoint/restore

## Demo Scripts

### Main Demo Scripts

1. **`run_complete_demo.sh`** - Complete end-to-end demo
   - Runs both baseline and checkpoint demos sequentially
   - Automatically cleans up between runs
   - Shows performance comparison
   - Usage: `./run_complete_demo.sh [prompt]`

2. **`demo_baseline_final.sh`** - Baseline cold start performance
   - Measures time from container start to first inference
   - Starts vLLM from scratch (no checkpoint)
   - Usage: `./demo_baseline_final.sh [prompt]`
   - Expected time: ~160 seconds

3. **`demo_checkpoint_final.sh`** - Checkpoint restore performance
   - Measures time from checkpoint restore to first inference
   - Requires checkpoint to exist first
   - Usage: `./demo_checkpoint_final.sh [prompt]`
   - Expected time: ~33 seconds

### Supporting Scripts

4. **`create_checkpoint.sh`** - Creates a checkpoint of fully loaded vLLM
   - Starts vLLM container with Qwen 32B
   - Sends warmup inference to compile CUDA graphs
   - Creates checkpoint using podman + CRIU + cuda-checkpoint
   - Usage: `./create_checkpoint.sh`
   - Time: ~2-3 minutes

5. **`checkpoint.sh`** - Legacy checkpoint restore script
   - Similar to demo_checkpoint_final.sh but without --keep flag
   - Checkpoint is consumed after restore

6. **`baseline.sh`** - Legacy baseline script
   - Similar to demo_baseline_final.sh
   - Older version kept for reference

## Documentation Files

### Performance Analysis

- **`FINAL_OPTIMIZATION_RESULTS.md`** - Complete optimization journey
  - Worker count optimization (64, 128, 256, 320, 512 workers tested)
  - Disk vs page cache analysis
  - Chunk size optimization (256MB vs 512MB)
  - Final performance results
  - Recommendations for future improvements

- **`DISK_VS_CACHE_ANALYSIS.md`** - Cold disk vs page cache performance
  - Identified disk I/O as bottleneck (4.2 GB/s from virtio storage)
  - Page cache performance: 5.9 GB/s
  - VMA restore timing breakdown

- **`WORKER_COUNT_ANALYSIS.md`** - VMA parallel worker optimization
  - Tested 64, 128, 256, 320, 512 workers
  - Found 256 workers optimal for 299 chunks
  - Explains parallelism vs overhead tradeoffs

- **`DEMO_README.md`** - Original demo setup instructions

## Quick Start

### Prerequisites

1. Lambda Labs instance (or similar) with H100 GPU
2. CRIU installed with cuda-checkpoint plugin
3. Podman container runtime
4. vLLM Docker image
5. Model cached locally: `Qwen/Qwen2.5-32B-Instruct`

### Run Complete Demo

```bash
cd /root/criu/demo_scripts
./run_complete_demo.sh "What is the capital of France?"
```

This will:
1. Clean up any existing containers
2. Run baseline cold start demo (~160s)
3. Clean up baseline container
4. Create checkpoint (~2 minutes)
5. Run checkpoint restore demo (~33s)
6. Clean up and show summary

### Run Individual Demos

**Baseline only:**
```bash
./demo_baseline_final.sh "Your prompt here"
```

**Checkpoint restore only:**
```bash
# Create checkpoint first (if not exists)
./create_checkpoint.sh

# Run restore demo
./demo_checkpoint_final.sh "Your prompt here"
```

## Performance Breakdown

### Baseline Cold Start (~160 seconds)
- Container startup: ~5s
- Model loading (17 shards): ~90s
- CUDA graph compilation: ~60s
- First inference: ~5s

### Checkpoint Restore (~33 seconds)
- CRIU VMA restore (86GB): ~20s
- GPU restore (76GB): ~14s
- Container coordination: ~2s
- First inference: immediate

## CRIU Optimizations Applied

### 1. VMA Parallelization
- Modified `/root/criu/criu/pie/restorer.c`
- Uses 256 parallel worker processes
- Each worker restores memory chunks independently
- Achieves 4.3 GB/s effective throughput (vs 0.76 GB/s sequential)

### 2. Chunk Size Optimization
- Modified `/root/criu/criu/cr-restore.c`
- Increased `max_iovec_mb` from 256MB to 512MB
- Reduced chunk count from 299 to 163
- Improved disk I/O efficiency by 8%

### 3. Per-Worker File Descriptors
- Each worker reopens pages file via /proc/self/fd/N
- Enables true parallel I/O (no file descriptor contention)

### Configuration

Current CRIU settings (hardcoded in cr-restore.c):
```c
opts.vma_parallel_workers = 256;
opts.max_iovec_mb = 512;
```

## Multi-GPU Support (Future Work)

For testing with larger models (e.g., GPT 120B on 4x H100):

**Requirements:**
- Multi-GPU instance (e.g., 4x H100)
- Add `--tensor-parallel-size 4` to vLLM
- No other changes needed (CRIU handles process trees automatically)

**Expected Performance:**
- GPT 120B restore: ~100-125 seconds
- Cold start: ~600-900 seconds
- Speedup: 5-8x faster

See multi-GPU analysis documents in `/tmp/` for details.

## Troubleshooting

### Checkpoint fails with OOM
- Another container is using GPU memory
- Solution: `podman rm -f $(podman ps -aq)`

### Restore fails with "checkpoint not found"
- Checkpoint was consumed by previous restore
- Solution: Recreate checkpoint with `./create_checkpoint.sh`
- Or: Use `--keep` flag in podman restore command

### Response parsing fails
- `jq` not installed
- Solution: `sudo apt-get install -y jq`

### Model loading takes >10 minutes
- Page cache was cleared (reading from cold disk)
- This is expected when testing cold disk performance
- Subsequent runs will be faster (warm cache)

## File Permissions

All scripts should be executable:
```bash
chmod +x *.sh
```

## Directory Structure

```
/root/criu/demo_scripts/
├── README.md                          # This file
├── run_complete_demo.sh               # Main demo script
├── demo_baseline_final.sh             # Baseline demo
├── demo_checkpoint_final.sh           # Checkpoint demo
├── create_checkpoint.sh               # Create checkpoint
├── checkpoint.sh                      # Legacy checkpoint script
├── baseline.sh                        # Legacy baseline script
├── FINAL_OPTIMIZATION_RESULTS.md      # Complete optimization journey
├── DISK_VS_CACHE_ANALYSIS.md          # Disk I/O analysis
├── WORKER_COUNT_ANALYSIS.md           # Worker optimization analysis
└── DEMO_README.md                     # Original demo instructions
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Model | Qwen 2.5 32B |
| GPU | 1x H100 (80GB) |
| Total Memory | 86GB (60GB weights + 26GB KV/graphs) |
| GPU Memory Used | 76GB |
| Baseline Time | 158.9s |
| Checkpoint Time | 127s |
| Restore Time | 33.1s |
| Speedup | 4.8x |
| VMA Restore | 20s @ 4.3 GB/s |
| GPU Restore | 14s |

## Future Optimizations

1. **NVMe Storage**: 7-10 GB/s (vs 4.2 GB/s virtio)
   - Expected improvement: 25-30% faster restore
   - VMA restore: 20s → 12-15s

2. **Larger max_iovec_mb**: Test 1024MB or 2048MB
   - Fewer chunks, better sequential reads
   - Marginal improvement expected

3. **Multi-GPU Testing**: Test TP=2 with Llama 3.1 70B
   - Validate NCCL state preservation
   - Measure actual multi-GPU restore time

## References

- CRIU: https://criu.org/
- cuda-checkpoint: NVIDIA HPC SDK
- vLLM: https://github.com/vllm-project/vllm
- Linux Kernel Coding Style (used in CRIU modifications)

## Authors

Optimizations and demo scripts created during GPU checkpoint/restore research on Lambda Labs infrastructure with CRIU v4.0 and vLLM v0.11.0.

Last updated: 2025-11-13
