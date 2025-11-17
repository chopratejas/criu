# GPU Checkpoint/Restore Setup Guide for vLLM with CRIU

Complete step-by-step guide for setting up GPU checkpointing with CRIU, NVIDIA CUDA plugin, and vLLM inference.

## System Information

- **OS**: Ubuntu 24.04 (Linux 6.11.0-29-generic)
- **GPU**: NVIDIA H100 PCIe (80GB VRAM)
- **RAM**: 226GB
- **Driver**: NVIDIA 570.158.01
- **Container Runtime**: Podman with runc
- **CRIU Version**: 4.0 (custom fork with VMA parallelization - 256 workers, 512MB chunks)

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [NVIDIA Driver Setup (580.95.05)](#nvidia-driver-setup)
3. [Building Custom CRIU](#building-custom-criu)
4. [Installing NVIDIA cuda-checkpoint](#installing-nvidia-cuda-checkpoint)
5. [Container Runtime Configuration](#container-runtime-configuration)
6. [NVIDIA Library Setup](#nvidia-library-setup)
7. [vLLM Container Setup](#vllm-container-setup)
8. [Checkpoint and Restore](#checkpoint-and-restore)
9. [Performance Results](#performance-results)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Packages

```bash
# Build dependencies for CRIU
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libnet-dev \
    protobuf-compiler \
    protobuf-c-compiler \
    libprotobuf-c-dev \
    libcap-dev \
    uuid-dev \
    python3-yaml \
    libaio-dev \
    git

# Container runtime
sudo apt-get install -y podman

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

---

## NVIDIA Driver Setup

The guide used NVIDIA Driver **580.95.05** which includes improved support for checkpointing.

### Verify Driver Installation

```bash
cat /proc/driver/nvidia/version
# Should show: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05
```

### Check GPU Devices

```bash
ls -la /dev/nvidia*
# Should show:
# /dev/nvidia0 (195:0) - GPU device
# /dev/nvidiactl (195:255) - Control device
# /dev/nvidia-uvm (234:0) - Unified Virtual Memory
# /dev/nvidia-uvm-tools (234:1)
# /dev/nvidia-modeset (195:254)
```

---

## Building Custom CRIU

### Clone Custom CRIU Fork

This fork includes VMA (Virtual Memory Area) parallelization optimizations with 256 workers (optimized) and 512MB chunk size for faster memory restoration.

**Latest optimizations** (as of 2025-11-17):
- **256 parallel workers**: Optimal for large checkpoints (86GB+)
- **512MB max_iovec_mb**: Reduces chunk count by 45% for better disk I/O
- **Per-worker file descriptors**: Independent I/O for true parallelism

```bash
cd /root
# Assume you have the custom CRIU fork at /root/criu
cd /root/criu
```

### Build CRIU

```bash
make clean
make -j$(nproc)
```

### Install CRIU Binary and CUDA Plugin

```bash
# Install CRIU binary
sudo cp criu/criu /usr/sbin/criu

# Install CUDA plugin
sudo mkdir -p /usr/lib/criu
sudo cp plugins/cuda/cuda_plugin.so /usr/lib/criu/cuda_plugin.so

# Verify installation
criu --version
# Should show: Version: 4.0, GitID: 4eda0ac94

# Verify CUDA plugin
ls -lah /usr/lib/criu/cuda_plugin.so
# Should be ~82K (custom version)
```

### Key Feature: VMA Parallelization

The custom CRIU includes hardcoded VMA workers in `/root/criu/criu/cr-restore.c`:

```c
/* HARDCODED: Force VMA parallel workers = 256 and max iovec size = 512MB */
opts.vma_parallel_workers = 256;
opts.max_iovec_mb = 512;
```

**Why these values:**
- **256 workers**: Optimal for 86GB+ checkpoints, creates ~1 chunk per worker for parallel I/O
- **512MB chunks**: Reduces chunk count from 299 to 163 (45% reduction), improves disk sequential reads by 8%

**Worker count optimization:**
| Workers | Chunks | VMA Restore | Result |
|---------|--------|-------------|--------|
| 64      | 318    | 28.7s       | Too few = sequential processing |
| 128     | 315    | 23.6s       | Still sequential |
| **256** | **299/163** | **20s** | **Optimal** - 1 chunk per worker |
| 320     | 322    | 16.5s (cached) | Overhead without benefit |
| 512     | N/A    | CRASH       | Segmentation fault |

To modify workers or chunk size:
1. Edit `/root/criu/criu/cr-restore.c` line 2430-2431
2. Edit `/root/criu/criu/pie/restorer.c` line 184, 2385-2397 (array sizes)
3. Rebuild: `make clean && make -j$(nproc)`
4. Reinstall: `sudo cp criu/criu /usr/sbin/criu && sudo cp plugins/cuda/cuda_plugin.so /usr/lib/criu/cuda_plugin.so`

---

## Installing NVIDIA cuda-checkpoint

**CRITICAL**: The mock cuda-checkpoint in `/root/criu/test/cuda-checkpoint/` does NOT work. You must use the real NVIDIA version.

### Download Real NVIDIA cuda-checkpoint

```bash
cd /root
git clone https://github.com/NVIDIA/cuda-checkpoint.git
```

### Install cuda-checkpoint Binary

```bash
sudo cp cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/cuda-checkpoint
sudo chmod +x /usr/local/bin/cuda-checkpoint

# Verify
ls -lah /usr/local/bin/cuda-checkpoint
# Should be ~5976 bytes (real NVIDIA version)
```

**Key Difference**:
- Mock version: 20KB, doesn't actually checkpoint GPU state
- Real version: ~6KB, properly handles GPU checkpoint/restore via NVIDIA driver

---

## Container Runtime Configuration

### 1. Configure Podman to Use runc

Create/edit `/etc/containers/containers.conf`:

```toml
[engine]
runtime = "runc"
```

### 2. Verify runc Version

```bash
runc --version
# Should show: runc version 1.2.5 or higher
```

### 3. Create seccomp Profile (Disable io_uring)

Create `/etc/containers/seccomp.d/no-io-uring.json`:

```json
{
  "defaultAction": "SCMP_ACT_ALLOW",
  "syscalls": [
    {
      "names": ["io_uring_setup", "io_uring_enter", "io_uring_register"],
      "action": "SCMP_ACT_ERRNO"
    }
  ]
}
```

**Why**: CRIU doesn't support io_uring syscalls, so we disable them for checkpointed containers.

### 4. Create CRIU Configuration for runc

Create `/etc/criu/runc.conf`:

```
# CRIU options for runc/Podman container checkpoint/restore
tcp-established
link-remap
work-dir /tmp
```

**Options explained**:
- `tcp-established`: Preserve TCP connections during checkpoint
- `link-remap`: Handle unlinked files by creating temporary hardlinks
- `work-dir /tmp`: Write CRIU logs to /tmp for debugging

### 5. Generate NVIDIA CDI Configuration

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

This allows podman to access GPUs via `--device nvidia.com/gpu=all`.

---

## NVIDIA Library Setup

### Create NVIDIA Libraries Directory

Some NVIDIA libraries need to be mounted into containers for proper GPU access:

```bash
sudo mkdir -p /opt/nvidia-libs

# Copy required libraries
sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so* /opt/nvidia-libs/
sudo cp /usr/lib/x86_64-linux-gnu/libnvidia-*.so* /opt/nvidia-libs/

# Verify
ls -lah /opt/nvidia-libs/ | grep libcuda
# Should show libcuda.so, libcuda.so.1, libcuda.so.580.95.05
```

**Libraries to include**:
- `libcuda.so*` - CUDA driver library
- `libnvidia-ml.so*` - NVML (management library)
- `libnvidia-nvvm.so*` - NVVM compiler
- `libnvidia-cfg.so*` - Configuration library

---

## vLLM Container Setup

### Environment Configuration

Create `/root/gpu-checkpoint-benchmarks/.env`:

```bash
# Container Configuration
CONT_NAME="vllm-checkpoint"
API_PORT="8000"

# Model Configuration - Qwen 7B
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN="4096"
GPU_MEMORY_UTIL="0.90"

# Model Configuration - Qwen 14B AWQ (Quantized)
# MODEL_ID="Qwen/Qwen2.5-14B-Instruct-AWQ"
# MAX_MODEL_LEN="8192"
# GPU_MEMORY_UTIL="0.90"

# Checkpoint Configuration
CHECKPOINT_DIR="/var/lib/containers/storage"
WARMUP_REQUESTS="5"

# Health Check
HEALTH_CHECK_TIMEOUT="300"
```

### Container Launch Command

**Exact command that works** (from create-checkpoint.py):

```bash
source /root/gpu-checkpoint-benchmarks/.env

podman run -d \
  --name vllm-checkpoint \
  --device /dev/null:/dev/null:rwm \
  --privileged \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --shm-size 8g \
  -e "LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH" \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --trust-remote-code \
  --load-format safetensors \
  --enforce-eager
```

**Key flags**:
- `--device /dev/nvidiactl`: REQUIRED despite causing issues - needed for GPU detection
- `--privileged`: Needed for full device access
- `--security-opt seccomp=...`: Disable io_uring
- `--device /dev/nvidia0, /dev/nvidia-uvm`: GPU compute devices
- `--shm-size 8g`: Shared memory for vLLM
- `-e ASYNCIO_DEFAULT_BACKEND=select`: Avoid io_uring in Python
- `-e PYTHON_ASYNCIO_NO_IO_URING=1`: Disable io_uring in asyncio
- `-v /opt/nvidia-libs:/opt/nvidia-libs:ro`: Mount NVIDIA libraries
- `--enforce-eager`: Disable cudagraph (better for checkpointing)

### For Qwen 14B-AWQ (Quantized):

```bash
podman run -d \
  --name vllm-checkpoint \
  --device /dev/null:/dev/null:rwm \
  --privileged \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --shm-size 8g \
  -e "LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH" \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code \
  --enforce-eager
```

### Wait for Container Ready

```bash
# Wait for health endpoint
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM is ready!"
```

---

## Checkpoint and Restore

### Create Checkpoint

```bash
podman container checkpoint vllm-checkpoint
```

**What happens**:
1. CRIU freezes the container processes
2. CUDA plugin calls `cuda-checkpoint` to save GPU state (VRAM → CPU memory)
3. CRIU saves process memory, file descriptors, network connections to disk
4. Container enters "Exited" state

**Checkpoint location**:
```
/var/lib/containers/storage/overlay-containers/<container-id>/userdata/checkpoint/
```

**Checkpoint size** (Qwen 7B): ~23GB
- 22GB: GPU VRAM (model weights)
- 1GB: Process memory, state, file descriptors

### Restore from Checkpoint

```bash
# Restore
time podman container restore vllm-checkpoint

# Verify container is running
podman ps | grep vllm

# Test inference
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "Hello", "max_tokens": 10}'
```

**What happens during restore**:
1. CRIU reads checkpoint images
2. Recreates process tree and namespaces
3. VMA workers (64 parallel) restore memory mappings
4. CUDA plugin calls `cuda-checkpoint` to restore GPU state (CPU memory → VRAM)
5. Container resumes execution

### Check CRIU Logs

```bash
# Restore log shows VMA workers and timing
cat /tmp/restore.log | grep "VMA PARALLEL"
# Output: VMA PARALLEL: Using 64 workers (hardcoded)

# Check GPU restore timing
tail -20 /tmp/restore.log
# Shows cuda_plugin timing summary
```

---

## Performance Results

### Latest Results: Qwen 32B on H100 (November 2025)

**Hardware**: 1x H100 PCIe (80GB), 226GB RAM, virtio storage (4.2 GB/s)
**Model**: Qwen 2.5 32B (~60GB weights, 86GB total memory)

| Phase | Cold Start | Checkpoint Restore | Speedup |
|-------|------------|-------------------|---------|
| **Total time to first inference** | 158.9s | **33.1s** | **4.8x** |
| Model loading | ~90s | - | - |
| CUDA graph compilation | ~60s | - | - |
| VMA restore (CPU memory) | - | 20s @ 4.3 GB/s | - |
| GPU restore | - | 14s | - |
| Container overhead | ~8s | ~2s | - |

**Key Findings**:
- **256 VMA workers + 512MB chunks** optimal for 86GB checkpoint
- Checkpoint restore is **4.8x faster** than cold start
- VMA parallelization achieves **4.3 GB/s** effective throughput (6x vs sequential)
- GPU restore is consistent at ~14s regardless of model size

### Disk vs Page Cache Performance

| Storage | VMA Restore | GPU Restore | Total | Throughput |
|---------|-------------|-------------|-------|------------|
| **Cold disk (virtio)** | 22.1s | 13.8s | **43.5s** | 3.9 GB/s |
| **Page cache (warm)** | 14.6s | 13.9s | **36.4s** | 5.9 GB/s |
| **With 512MB chunks (disk)** | 20.3s | 13.8s | **41.7s** | 4.2 GB/s |

**Bottleneck**: Disk I/O is the primary bottleneck. With NVMe (7-10 GB/s), VMA restore could improve from 20s to 12-15s.

### Worker Count Optimization (Qwen 32B)

| Workers | Chunks | VMA Restore (disk) | Result |
|---------|--------|--------------------|--------|
| 64      | 318    | 28.7s              | Sequential (5 chunks/worker) |
| 128     | 315    | 23.6s              | Sequential (2.5 chunks/worker) |
| **256** | **299** | **22.1s** (256MB) | **Optimal** (1.2 chunks/worker) |
| **256** | **163** | **20.3s** (512MB) | **Best** (0.6 chunks/worker) |
| 320     | 322    | 16.5s (cached)     | Overhead without benefit |
| 512     | N/A    | CRASH              | Segmentation fault |

**Conclusion**: 256 workers with 512MB chunks is optimal for large models (60GB+)

### Chunk Size Impact

| max_iovec_mb | Chunks | VMA Restore | Checkpoint Time | Speedup |
|--------------|--------|-------------|-----------------|---------|
| 256MB        | 299    | 22.1s       | 133.9s          | Baseline |
| **512MB**    | **163** | **20.3s** | **124.8s** | **8% faster** restore, 7% faster checkpoint |

**Benefits of 512MB chunks**:
- 45% fewer chunks (better for disk seeks)
- Larger sequential reads (better for virtio storage)
- Faster both checkpoint and restore

### Model Comparison (Custom CRIU, 256 workers, 512MB chunks)

| Model | Size | GPU Memory | Total Restore (disk) | Cold Start | Speedup |
|-------|------|------------|---------------------|------------|---------|
| Qwen 7B | ~14GB | 23GB | ~13-15s (estimated) | ~80-100s | ~6-7x |
| Qwen 32B | ~60GB | 76GB | **33.1s** | **158.9s** | **4.8x** |
| Llama 70B (TP=2)* | ~140GB | 150GB | ~50-70s (estimated) | ~400-500s | ~7-8x |
| GPT 120B (TP=4)* | ~240GB | 300GB | ~100-125s (estimated) | ~600-900s | ~5-8x |

*Multi-GPU estimates based on scaling analysis

---

## Multi-GPU Support (Tensor Parallelism)

### Overview

For larger models that don't fit on a single GPU, vLLM supports Tensor Parallelism (TP) which splits the model across multiple GPUs. CRIU + cuda-checkpoint should support multi-GPU checkpointing, though this requires testing to confirm.

### Setup for 2+ GPUs

**Hardware Requirements:**
- 2x H100 (160GB total) for Llama 3.1 70B
- 4x H100 (320GB total) for GPT 120B
- Single node (all GPUs on same NVLink/PCIe fabric)

**vLLM Configuration:**
Add `--tensor-parallel-size N` flag:

```bash
podman run -d \
  --name vllm-70b-test \
  --device nvidia.com/gpu=all \
  --security-opt=seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --security-opt=label=disable \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:v0.11.0 \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --dtype auto
```

### How Multi-GPU Checkpointing Works

**Process Architecture:**
```
vLLM Master Process
├── GPU Worker 0 (1/N of model)
├── GPU Worker 1 (1/N of model)
├── GPU Worker 2 (1/N of model)
└── GPU Worker 3 (1/N of model)
     └── NCCL coordinates between workers
```

**Checkpoint Process:**
1. CRIU checkpoints entire process tree (master + all workers)
2. cuda-checkpoint called for EACH GPU worker in parallel
3. All GPU states saved atomically
4. NCCL communication state preserved (or re-initialized on restore)

**Restore Process:**
1. CRIU restores process tree
2. VMA workers restore CPU memory (256 workers across ALL memory)
3. All GPU workers restore in parallel
4. NCCL re-initialization (~10-20s if state not preserved)

### Expected Performance (Estimated)

**Llama 3.1 70B on 2x H100:**
- Memory: ~250GB total (system RAM)
- GPU memory: ~150GB across 2 GPUs
- Checkpoint time: ~180-220s
- Restore time: **50-70s** (estimated)
  - VMA restore: ~35-45s (250GB)
  - GPU restore: ~12-18s (parallel across 2 GPUs)
  - NCCL re-init: ~5-10s (if needed)
- Cold start: ~400-500s
- **Speedup: 7-8x**

**GPT 120B on 4x H100:**
- Memory: ~400-500GB total (system RAM)
- GPU memory: ~300GB across 4 GPUs
- Checkpoint time: ~135-160s
- Restore time: **100-125s** (estimated)
  - VMA restore: ~75-85s (400GB)
  - GPU restore: ~18-22s (parallel across 4 GPUs)
  - NCCL re-init: ~15-25s (if needed)
- Cold start: ~600-900s
- **Speedup: 5-8x**

### Recommended Testing Path

1. **Start with 2 GPUs (TP=2):**
   - Use Llama 3.1 70B or Qwen 2.5 72B
   - Verify checkpoint/restore works with multiple GPU workers
   - Measure if NCCL needs re-initialization
   - Validate performance scaling

2. **Scale to 4 GPUs (TP=4):**
   - Test with GPT 70B or similar large model
   - Confirm parallel GPU restore
   - Check for any sequential bottlenecks

3. **Key metrics to capture:**
   - Per-GPU checkpoint/restore time (should be parallel)
   - NCCL re-init time (if any)
   - Total restore time vs cold start
   - Inference quality after restore

### Potential Issues

**NCCL State Preservation:**
- Unknown if cuda-checkpoint preserves NCCL communication state
- If not preserved: Need 10-30s for NCCL re-initialization
- Still much faster than cold start

**GPU Topology:**
- Must restore on same physical GPU IDs
- NVLink topology should match checkpoint environment
- CUDA graphs may have hardcoded GPU references

### No Special Configuration Needed

**Good news**: Your existing CRIU setup should work with multi-GPU!
- No changes to CRIU configuration
- No changes to cuda-checkpoint configuration
- Same 256 workers, 512MB chunks
- Just add `--tensor-parallel-size N` to vLLM

---

## Demo Scripts

Complete demo scripts are available in `/root/criu/demo_scripts/`:

- **`run_complete_demo.sh`**: End-to-end baseline vs checkpoint comparison
- **`demo_baseline_final.sh`**: Cold start performance measurement
- **`demo_checkpoint_final.sh`**: Checkpoint restore performance measurement
- **`create_checkpoint.sh`**: Creates checkpoint of fully loaded vLLM

**Documentation:**
- `README.md`: Complete guide to demo scripts
- `FINAL_OPTIMIZATION_RESULTS.md`: Optimization journey and results
- `DISK_VS_CACHE_ANALYSIS.md`: Disk I/O bottleneck analysis
- `WORKER_COUNT_ANALYSIS.md`: Worker count optimization testing

**Usage:**
```bash
cd /root/criu/demo_scripts
./run_complete_demo.sh "What is the capital of France?"
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "Can't dump file 20 of that type (chr 195:255)"

**Symptom**: Checkpoint fails with error about /dev/nvidiactl

**Root Cause**: Using **mock cuda-checkpoint** instead of real NVIDIA version

**Solution**:
```bash
# Remove mock version
rm /usr/local/bin/cuda-checkpoint

# Install real NVIDIA version
git clone https://github.com/NVIDIA/cuda-checkpoint.git
sudo cp cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/
sudo chmod +x /usr/local/bin/cuda-checkpoint

# Verify size (~6KB, not 20KB)
ls -lah /usr/local/bin/cuda-checkpoint
```

#### 2. "handle_device_vma plugin failed"

**Symptom**: PyTorch/vLLM checkpoint fails on GPU memory mappings

**Cause**: cuda-checkpoint not in PATH or wrong version

**Solution**:
```bash
# Verify cuda-checkpoint is accessible
which cuda-checkpoint
# Should show: /usr/local/bin/cuda-checkpoint

# Test it works
cuda-checkpoint --help
```

#### 3. "Plugin cuda_plugin has 13 assigned while max 12 supported"

**Symptom**: Custom cuda_plugin.so incompatible with PPA CRIU

**Cause**: Mixing custom CRIU with vanilla plugin (or vice versa)

**Solution**: Always use matching CRIU + cuda_plugin.so:
```bash
# Custom CRIU + Custom plugin (82K)
sudo cp /root/criu/criu/criu /usr/sbin/criu
sudo cp /root/criu/plugins/cuda/cuda_plugin.so /usr/lib/criu/cuda_plugin.so

# OR Vanilla CRIU + Vanilla plugin (27K)
sudo apt-get install criu=4.1.1-2ppa1.24.04
# Plugin comes with package
```

#### 4. vLLM fails to detect GPU without --privileged

**Symptom**: "Failed to infer device type" or "No platform detected"

**Cause**: Insufficient permissions without --privileged flag

**Solution**: Use `--privileged` flag (required for GPU checkpoint/restore):
```bash
podman run -d --privileged --device /dev/nvidia0 ...
```

#### 5. Container won't start after restore

**Symptom**: Container shows "Exited" state after restore attempt

**Possible causes**:
- CRIU logs in /tmp/restore.log show errors
- GPU driver version mismatch
- Missing /dev/nvidiactl device

**Debug steps**:
```bash
# Check CRIU restore log
cat /tmp/restore.log | tail -50

# Verify GPU devices present
ls -la /dev/nvidia*

# Check if cuda_plugin loaded
grep "cuda_plugin" /tmp/restore.log
```

#### 6. Slow restore times (>30 seconds)

**Symptom**: Restore takes much longer than expected

**Likely cause**: Using vanilla CRIU without VMA parallelization

**Solution**: Build and install custom CRIU fork with VMA workers:
```bash
cd /root/criu
# Edit cr-restore.c to set workers = 64
make clean && make -j$(nproc)
sudo cp criu/criu /usr/sbin/criu
sudo cp plugins/cuda/cuda_plugin.so /usr/lib/criu/cuda_plugin.so
```

---

## Key Learnings

### What Works

✅ **Driver 570.158.01** (and 580.95.05) with CRIU 4.0 custom + real NVIDIA cuda-checkpoint
✅ **256 VMA workers + 512MB chunks** optimal for large checkpoints (60GB+)
✅ **--privileged** mode required for GPU access
✅ **--device nvidia.com/gpu=all** exposes all GPUs to container
✅ **Checkpoint restore 4.8x faster** than cold start for Qwen 32B
✅ **Sub-linear scaling**: Memory increases 5x but restore only 3-4x slower (parallelization works!)
✅ **vLLM with CUDA graphs** checkpoints reliably (no need for --enforce-eager)

### What Doesn't Work

❌ Mock cuda-checkpoint from /root/criu/test/cuda-checkpoint/
❌ Running without --privileged flag
❌ Vanilla CRIU without VMA parallelization (6x slower)
❌ Too few workers (64, 128): Sequential processing, 2-5x slower
❌ Too many workers (512): Segmentation fault
❌ Small chunk sizes (256MB): More disk seeks, 8% slower than 512MB

### Optimization Discoveries

**Worker Count:**
- Optimal when workers ≈ chunk count (1 chunk per worker)
- For 86GB checkpoint: 256 workers creates 163-299 chunks (optimal)
- Too few workers → sequential processing per worker
- Too many workers → overhead without benefit

**Chunk Size:**
- 512MB chunks reduce count by 45% (299 → 163)
- Larger sequential reads = better disk I/O efficiency
- Benefits both checkpoint creation (7% faster) and restore (8% faster)

**Disk I/O is the Bottleneck:**
- VMA restore: 20s (CPU memory I/O)
- GPU restore: 14s (GPU memory I/O)
- With NVMe (7-10 GB/s): Could reduce VMA restore to 12-15s
- Page cache performance: 40% faster than cold disk

### Critical Files Summary

| File | Purpose | Location |
|------|---------|----------|
| **criu binary** | Custom with 64 VMA workers | `/usr/sbin/criu` |
| **cuda_plugin.so** | CRIU plugin for GPU (82K custom) | `/usr/lib/criu/cuda_plugin.so` |
| **cuda-checkpoint** | Real NVIDIA utility (6KB) | `/usr/local/bin/cuda-checkpoint` |
| **runc.conf** | CRIU options for runc/podman | `/etc/criu/runc.conf` |
| **seccomp profile** | Disable io_uring | `/etc/containers/seccomp.d/no-io-uring.json` |
| **containers.conf** | Set runtime to runc | `/etc/containers/containers.conf` |
| **nvidia.yaml** | CDI config for GPU access | `/etc/cdi/nvidia.yaml` |
| **NVIDIA libraries** | Mounted into container | `/opt/nvidia-libs/` |

---

## References

- **Nilesh Agarwal's Blog**: [GPU Snapshots for Reducing ML Inference Cold Starts](https://nilesh-agarwal.com/gpu-snapshots-for-reducing-ml-inference-cold-starts-2/)
- **CRIU Official**: https://criu.org/
- **NVIDIA cuda-checkpoint**: https://github.com/NVIDIA/cuda-checkpoint
- **vLLM Documentation**: https://docs.vllm.ai/

---

## Version Information

| Component | Version | Notes |
|-----------|---------|-------|
| CRIU | 4.0 (GitID: 9a4ff277a) | Custom fork with 256 workers, 512MB chunks |
| cuda_plugin.so | 82KB (custom) | Built from custom CRIU fork |
| cuda-checkpoint | ~6KB | Real NVIDIA version from github.com/NVIDIA/cuda-checkpoint |
| NVIDIA Driver | 570.158.01 | Open Kernel Module (also tested: 580.95.05) |
| GPU | H100 PCIe (80GB) | Also tested: A10 (24GB) |
| runc | 1.2.5 | OCI runtime |
| Podman | Latest | Container engine |
| Ubuntu | 24.04 | Linux 6.11.0-29-generic |
| vLLM | 0.11.0 | From docker.io/vllm/vllm-openai:v0.11.0 |
| Model (tested) | Qwen 2.5 32B | 60GB weights, 86GB total memory |

---

## Quick Start Summary

```bash
# 1. Build custom CRIU
cd /root/criu && make -j$(nproc)
sudo cp criu/criu /usr/sbin/criu
sudo cp plugins/cuda/cuda_plugin.so /usr/lib/criu/cuda_plugin.so

# 2. Install real NVIDIA cuda-checkpoint
git clone https://github.com/NVIDIA/cuda-checkpoint.git
sudo cp cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/
sudo chmod +x /usr/local/bin/cuda-checkpoint

# 3. Setup configs
sudo mkdir -p /etc/criu /etc/containers/seccomp.d /opt/nvidia-libs

# Create runc.conf
echo "tcp-established
link-remap
work-dir /tmp" | sudo tee /etc/criu/runc.conf

# Create seccomp profile
echo '{
  "defaultAction": "SCMP_ACT_ALLOW",
  "syscalls": [{"names": ["io_uring_setup", "io_uring_enter", "io_uring_register"], "action": "SCMP_ACT_ERRNO"}]
}' | sudo tee /etc/containers/seccomp.d/no-io-uring.json

# Set runc runtime
echo "[engine]
runtime = \"runc\"" | sudo tee /etc/containers/containers.conf

# Copy NVIDIA libs
sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so* /opt/nvidia-libs/
sudo cp /usr/lib/x86_64-linux-gnu/libnvidia-*.so* /opt/nvidia-libs/

# Generate CDI
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 4. Launch vLLM (Qwen 32B example)
podman run -d \
  --name vllm-checkpoint \
  --device nvidia.com/gpu=all \
  --security-opt=seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --security-opt=label=disable \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:v0.11.0 \
  --model Qwen/Qwen2.5-32B-Instruct \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --dtype auto

# Wait for ready (this will take ~2-3 minutes for model load + CUDA compilation)
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done

# 5. Checkpoint
podman container checkpoint vllm-checkpoint
# Expected: ~120-135 seconds for Qwen 32B

# 6. Restore
time podman container restore vllm-checkpoint
# Expected: ~33 seconds for Qwen 32B (cold disk with 256 workers, 512MB chunks)

# 7. Test inference
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-32B-Instruct", "prompt": "What is the capital of France?", "max_tokens": 50}' | jq -r '.choices[0].text'
```

---

**Success Criteria**: When restore completes in ~33 seconds with logs showing "VMA PARALLEL: Using 256 workers" and "MAX IOVEC: Using 512 MB limit" and vLLM inference working immediately after restore.

**Check logs:**
```bash
grep "VMA PARALLEL\|MAX IOVEC" /tmp/restore.log
# Should show:
# VMA PARALLEL: Using 256 workers (hardcoded)
# MAX IOVEC: Using 512 MB limit (hardcoded)
```
