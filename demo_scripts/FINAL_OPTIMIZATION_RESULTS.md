# GPU Checkpoint/Restore Final Optimization Results

## Complete Test Results Summary (Qwen 32B, 86GB checkpoint)

| Config | Chunks | Workers | VMA Restore | GPU Restore | Total | I/O Speed | Status |
|--------|--------|---------|-------------|-------------|-------|-----------|--------|
| **256MB (cache)** | 299 | 256 | 14.6s | 13.9s | **36.4s** | 5.9 GB/s | ✓ Baseline |
| **256MB (disk)** | 299 | 256 | 22.1s | 13.8s | 43.5s | 3.9 GB/s | ✓ Cold disk |
| **512MB (disk)** | 163 | 160 | 20.3s | 13.8s | **41.7s** | 4.2 GB/s | ✓ **BEST** |

## Key Findings

### 1. **512MB chunks provide 8% faster disk I/O**
- **256MB chunks**: 299 chunks, 22.1s VMA restore, 3.9 GB/s
- **512MB chunks**: 163 chunks, 20.3s VMA restore, 4.2 GB/s
- **Improvement**: 1.8 seconds faster (4.1% total, 7.8% VMA only)

### 2. **Chunk count reduction**
- 299 → 163 chunks (**45% reduction**)
- Larger sequential reads = better disk efficiency
- Fewer disk seeks

### 3. **Automatic worker optimization**
CRIU intelligently reduces workers to match chunk count:
- 256MB: 299 chunks → uses all 256 workers
- 512MB: 163 chunks → automatically uses only 160 workers
- This prevents worker overhead when chunks < workers

### 4. **Page cache vs disk performance gap**
- **Page cache**: 14.6s VMA (5.9 GB/s)
- **Cold disk**: 20.3s VMA (4.2 GB/s)
- **Gap**: 5.9s difference (40% slower from disk)

## Detailed Breakdown

### 512MB Chunks (from /tmp/restore.log):
```
(00.595311) Computed 163 chunks for 160 workers
(00.595332) Spawning 160 workers...
(20.935718) All 160 workers reaped. PIDs freed for reuse.
VMA restore: 20.34 seconds (4.23 GB/s)

(22.000341) cuda_plugin: RESUME_DEVICES_LATE
(35.841197) cuda_plugin: RESUME_DEVICE completed: 13840806 us
GPU restore: 13.84 seconds
```

### 256MB Chunks (from previous test):
```
Computed ~299 chunks for 256 workers
Spawning 256 workers...
All 256 workers reaped. PIDs freed for reuse.
VMA restore: 22.06 seconds (3.90 GB/s)

GPU restore: 13.82 seconds
```

## Performance vs Baseline (Cold Disk)

| Configuration | vs 512MB | Speedup |
|---------------|----------|---------|
| 256MB chunks  | +1.8s (+4.3%) | 0.96x |
| **512MB chunks** | **Baseline** | **1.00x** |

## Checkpoint Creation Time

| Config | Checkpoint Time | Notes |
|--------|-----------------|-------|
| 256MB  | 133.9s | More chunks to create |
| 512MB  | 124.8s | Fewer, larger chunks |

**9.1 seconds faster checkpoint** with 512MB chunks!

## Complete Optimization Journey

### Phase 1: Worker Count Optimization
| Workers | VMA Restore | Result |
|---------|-------------|--------|
| 64      | 28.7s       | Too few = sequential processing |
| 128     | 23.6s       | Still sequential |
| **256** | **14.6s**   | **Optimal for 299 chunks** |
| 320     | 16.5s       | Overhead |
| 512     | CRASH       | Segmentation fault |

**Learning**: Workers should match or slightly exceed chunk count for optimal parallelism.

### Phase 2: Disk vs Cache Discovery
- Discovered all tests were reading from page cache
- Cold disk is 40% slower than cache (5.9 → 4.2 GB/s)
- **Disk I/O is the real bottleneck**

### Phase 3: Chunk Size Optimization
- Increased max_iovec_mb from 256MB → 512MB
- Reduced chunks by 45% (299 → 163)
- Improved disk throughput by 8% (3.9 → 4.2 GB/s)
- **1.8 seconds faster restore from cold disk**

## Final Recommendation

**Use max_iovec_mb=512MB with 256 workers** because:

1. **Faster restore**: 41.7s vs 43.5s from cold disk (4% improvement)
2. **Faster checkpoint**: 124.8s vs 133.9s (7% improvement)
3. **Better disk efficiency**: 4.23 GB/s vs 3.90 GB/s
4. **Fewer chunks**: 163 vs 299 (less overhead)
5. **Auto-optimized workers**: CRIU uses 160 workers (matches chunk count)

## Performance Summary Table

| Metric | 256MB Chunks | 512MB Chunks | Improvement |
|--------|--------------|--------------|-------------|
| **Chunk Count** | 299 | 163 | -45% |
| **Workers Used** | 256 | 160 | Auto-optimized |
| **VMA Restore** | 22.1s | 20.3s | -1.8s (-8%) |
| **Total Restore** | 43.5s | 41.7s | -1.8s (-4%) |
| **I/O Throughput** | 3.9 GB/s | 4.2 GB/s | +8% |
| **Checkpoint Time** | 133.9s | 124.8s | -9.1s (-7%) |

## Next Optimization Opportunities

### 1. Storage Hardware Upgrade
- Current: virtio (4.2 GB/s)
- NVMe: Could achieve 7-10 GB/s
- **Potential**: Reduce VMA restore from 20s to ~8-12s

### 2. Further Increase max_iovec_mb
- Current: 512MB
- Test: 1024MB or 2048MB
- May further reduce chunk count and improve sequential reads

### 3. I/O Optimization Flags
- O_DIRECT: Bypass page cache overhead
- io_uring: Async I/O for better parallelism
- MADV_SEQUENTIAL: Better kernel read-ahead

### 4. Pre-warming Page Cache
- For production: Keep checkpoint in page cache
- Restore in 36.4s (vs 41.7s cold disk)
- Trade memory for speed

## Conclusion

**We achieved significant optimizations through this journey:**

1. **Worker count optimization**: Found 256 workers optimal for our chunk size
2. **Disk vs cache understanding**: Identified true bottleneck
3. **Chunk size optimization**: 512MB chunks provide best balance

**Final Configuration:**
- **Workers**: 256 (auto-reduces to 160 based on chunk count)
- **max_iovec_mb**: 512MB
- **Cold disk restore**: 41.7 seconds (4.2 GB/s)
- **Page cache restore**: ~36 seconds (5.9 GB/s)

**Total improvement from initial 256MB cold disk**: 1.8 seconds (4.1% faster)

---

*Model: Qwen/Qwen2.5-32B-Instruct (60GB)*  
*Hardware: H100 GPU, 26 CPUs, 226GB RAM*  
*Storage: virtio block device*  
*CRIU: v4.0 with VMA parallelization*
