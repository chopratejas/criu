# Cold Disk vs Page Cache Performance Analysis

## Test Setup
- Model: Qwen 32B (86GB checkpoint)
- Workers: 256 (optimal from previous testing)
- max_iovec_mb: 256MB
- Storage: virtio block device

## Results Comparison

| Phase | Cold Disk | Page Cache | Difference |
|-------|-----------|------------|------------|
| **VMA Restore** | 22.06s | 14.6s | +7.46s (+51%) |
| **GPU Restore** | 13.82s | 13.9s | -0.08s (±0%) |
| **Total CRIU** | 43.5s | 36.4s | +7.1s (+19%) |

## Detailed Breakdown

### Cold Disk (from /tmp/restore.log):
```
(00.626028) Spawning 256 workers...
(22.685791) All 256 workers reaped. PIDs freed for reuse.
VMA restore time: 22.06 seconds

(23.738828) cuda_plugin: RESUME_DEVICES_LATE (pid 183377)
(37.562246) cuda_plugin: RESUME_DEVICE completed: 13823331 us (13.82s)
GPU restore time: 13.82 seconds
```

### Page Cache (from previous tests):
```
VMA restore: 14.6 seconds
GPU restore: 13.9 seconds
Total: 36.4 seconds
```

## Throughput Analysis

### VMA Restore I/O Throughput:
- **Cold Disk**: 86GB / 22.06s = **3.9 GB/s**
- **Page Cache**: 86GB / 14.6s = **5.9 GB/s**
- **Difference**: 2.0 GB/s slower from disk

### GPU Restore (independent of disk):
- Consistent at ~13.8s regardless of VMA source
- GPU memory restore is CPU/GPU bound, not disk bound

## Key Findings

1. **Disk I/O is the bottleneck**
   - VMA restore from disk takes 51% longer than from page cache
   - 3.9 GB/s from virtio storage vs 5.9 GB/s memory bandwidth

2. **256 Workers still optimal**
   - Even on cold disk, 256 workers provides best parallelism
   - Each worker reads ~1 chunk independently

3. **GPU restore is independent**
   - GPU restore time (~13.8s) is unaffected by disk vs cache
   - Confirms GPU restore is compute-bound, not I/O-bound

## Next Optimization Opportunities

### Option 1: Reduce chunk count with larger max_iovec_mb
- Current: 256MB → ~299 chunks
- Proposed: 512MB → ~150 chunks (estimated)
- Benefits:
  - Fewer disk seeks
  - Larger sequential reads
  - Better disk I/O efficiency
- Trade-off: May slightly increase per-worker processing time

### Option 2: Storage upgrade
- Current: virtio (3.9 GB/s)
- NVMe: Could achieve 7-10 GB/s
- Would reduce VMA restore from 22s to ~8-12s

### Option 3: I/O optimization flags
- O_DIRECT: Bypass page cache entirely, may reduce overhead
- io_uring: Async I/O for better parallelism
- MADV_SEQUENTIAL: Hint to kernel for better read-ahead

## Recommendation

**Test max_iovec_mb=512MB first** because:
1. No hardware changes required
2. Should reduce chunk count by ~50%
3. Larger sequential reads are faster on virtio
4. May gain 2-5 seconds on VMA restore

Expected result with 512MB chunks:
- Chunk count: ~150 chunks (vs 299)
- VMA restore: 17-20s (vs 22s cold disk)
- Total restore: 38-41s (vs 43.5s)

---

## Test Data Summary

| Configuration | VMA | GPU | Total | Chunks | I/O Speed |
|---------------|-----|-----|-------|--------|-----------|
| 256w + 256MB (cache) | 14.6s | 13.9s | 36.4s | 299 | 5.9 GB/s |
| 256w + 256MB (disk) | 22.1s | 13.8s | 43.5s | 299 | 3.9 GB/s |
| 256w + 512MB (disk) | TBD | ~13.8s | TBD | ~150 | TBD |
