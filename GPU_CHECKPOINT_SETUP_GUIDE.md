# GPU Checkpoint/Restore Setup Guide for vLLM with CRIU

Complete step-by-step guide for setting up GPU checkpointing with CRIU, NVIDIA CUDA plugin, and vLLM inference.

## System Information

- **OS**: Ubuntu 24.04 (Linux 6.11.0-29-generic)
- **GPU**: NVIDIA A10 (24GB VRAM)
- **RAM**: 222GB
- **Driver**: NVIDIA 580.95.05
- **Container Runtime**: Podman with runc
- **CRIU Version**: 4.0 (custom fork with VMA parallelization)

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

This fork includes VMA (Virtual Memory Area) parallelization optimizations with 64 workers (default) for faster memory restoration.

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
/* HARDCODED: Force VMA parallel workers = 64 and max iovec size = 256MB */
opts.vma_parallel_workers = 64;
opts.max_iovec_mb = 256;
```

To modify workers (e.g., 256):
1. Edit `/root/criu/criu/cr-restore.c` line 2430
2. Rebuild: `make clean && make -j$(nproc)`
3. Reinstall: `sudo cp criu/criu /usr/sbin/criu && sudo cp plugins/cuda/cuda_plugin.so /usr/lib/criu/cuda_plugin.so`

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

### Custom CRIU 4.0 vs Vanilla CRIU 4.1.1

**Test**: Qwen 7B (14.25 GiB model, 23GB checkpoint)

| Metric | Custom CRIU 4.0 (64 workers) | Vanilla CRIU 4.1.1 (no workers) | Speedup |
|--------|------------------------------|----------------------------------|---------|
| **Total restore time** | 12.78s | 27.36s | **2.14x** |
| **CRIU restore** | 8.46s | 22.91s | **2.71x** |
| GPU restore | 6.74s | 6.34s | Similar |
| VMA/memory restore | 1.72s | 16.57s | **9.64x** |
| Podman overhead | 4.32s | 4.45s | Similar |

**Key Findings**:
- **64 VMA workers provide 9.6x speedup** for memory restoration
- GPU restore dominates (~50-80% of time) regardless of version
- Custom CRIU delivers **2.14x faster total restore** for large model checkpoints

### Model Comparison (Custom CRIU, 64 workers)

| Model | Size | Total Restore | CRIU Time | GPU Restore | VMA Time |
|-------|------|---------------|-----------|-------------|----------|
| Qwen 7B | 14.25 GiB | 12.78s | 8.46s | 6.74s | 1.72s |
| Qwen 14B-AWQ | 9.38 GiB | 13.11s | 8.57s | 6.72s | 1.85s |

**Observation**: Restore time is similar because:
- Quantized 14B (9.4GB) is smaller than full 7B (14.25GB)
- GPU restore dominates and is consistent across models
- VMA parallelization efficiently handles memory differences

### VMA Worker Scaling (Qwen 14B-AWQ)

| Workers | Total Time | CRIU Time | GPU Restore | VMA Time |
|---------|-----------|-----------|-------------|----------|
| 64 | 13.11s | 8.57s | 6.72s | 1.85s |
| 256 | 13.79s | 9.18s | 7.38s | 1.80s |

**Finding**: **64 workers is optimal** for this workload. 256 workers add overhead without benefit, likely due to:
- Increased context switching
- More complex synchronization
- Diminishing returns for 24GB checkpoint size

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

✅ **Driver 580.95.05** with CRIU 4.0 custom + real NVIDIA cuda-checkpoint
✅ **64 VMA workers** optimal for 20-30GB checkpoints
✅ **--privileged** mode required for GPU access
✅ **--device /dev/nvidiactl** must be included despite causing issues in older setups
✅ **Quantized models** (AWQ) checkpoint just as fast as full models
✅ **vLLM with --enforce-eager** works reliably for checkpoint/restore

### What Doesn't Work

❌ Mock cuda-checkpoint from /root/criu/test/cuda-checkpoint/
❌ Running without --privileged flag
❌ Vanilla CRIU without VMA parallelization (2.7x slower)
❌ More than 64 workers (diminishing returns, adds overhead)
❌ Removing /dev/nvidiactl from container devices

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
| CRIU | 4.0 (GitID: 4eda0ac94) | Custom fork with VMA parallelization |
| cuda_plugin.so | 82KB (custom) | Built from custom CRIU fork |
| cuda-checkpoint | ~6KB | Real NVIDIA version |
| NVIDIA Driver | 580.95.05 | Open Kernel Module |
| runc | 1.2.5 | OCI runtime |
| Podman | Latest | Container engine |
| Ubuntu | 24.04 | Linux 6.11.0-29-generic |
| vLLM | 0.11.0 | From docker.io/vllm/vllm-openai:latest |

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

# 4. Launch vLLM
podman run -d \
  --name vllm-checkpoint \
  --device /dev/null:/dev/null:rwm \
  --privileged \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --shm-size 8g \
  -e LD_LIBRARY_PATH=/opt/nvidia-libs \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --trust-remote-code --load-format safetensors --enforce-eager

# Wait for ready
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done

# 5. Checkpoint
podman container checkpoint vllm-checkpoint

# 6. Restore
time podman container restore vllm-checkpoint
# Expected: ~12-13 seconds for 7B/14B-AWQ models
```

---

**Success Criteria**: When restore completes in ~12-13 seconds with logs showing "VMA PARALLEL: Using 64 workers" and vLLM inference working immediately after restore.
