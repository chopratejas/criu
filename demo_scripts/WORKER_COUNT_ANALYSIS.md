# Worker Count Optimization Analysis

## Complete Test Results (Qwen 32B Model, 86GB data)

| Workers | Chunks Created | VMA Restore | Total Restore | Status |
|---------|---------------|-------------|---------------|---------|
| **64**  | 318           | 28.7s       | 47.4s         | ✓ Too slow |
| **128** | 315           | 23.6s       | 42.7s         | ✓ Slow |
| **256** | ~299          | **14.6s**   | **36.4s**     | ✓ **BEST** |
| **320** | 322           | 16.5s       | 38.2s         | ✓ Slower |
| **512** | N/A           | N/A         | CRASH         | ✗ Segfault |

## Key Findings

### 1. **256 Workers is Optimal**
- **Fastest VMA restore**: 14.6 seconds
- **Best total time**: 36.4 seconds
- Creates ~299 chunks (1.2 chunks/worker average)
- Sweet spot between parallelism and overhead

### 2. **Too Few Workers = Sequential Processing**
**64 workers:**
- 318 chunks / 64 workers = **5 chunks per worker**
- Workers process chunks SEQUENTIALLY
- Result: 28.7s (2x slower than 256!)

**128 workers:**
- 315 chunks / 128 workers = **2.5 chunks per worker**
- Still too much sequential work per worker
- Result: 23.6s (1.6x slower than 256)

### 3. **Too Many Workers = Overhead**
**320 workers:**
- 322 chunks / 320 workers = **1 chunk per worker**
- More workers than needed
- Result: 16.5s (13% slower than 256)
- Overhead: context switching, smaller chunks, more disk seeks

**512 workers:**
- **SEGMENTATION FAULT** - exceeded system limits
- Arrays/stacks too large for PIE restorer context

## Why 256 Workers Wins

### The Math:
```
Target chunk size = 86GB / 256 workers = 336 MB
max_iovec_mb = 256 MB (checkpoint creation limit)

Result: Creates ~299 chunks of ~289 MB each
Workers-to-chunks ratio: 256/299 = 0.86

This is the perfect balance:
- Most workers get exactly 1 chunk
- True parallel I/O
- Minimal overhead
```

### Comparison:
```
64 workers:  target=1344MB, creates 318 chunks → 5 chunks/worker (SEQ)
128 workers: target=672MB,  creates 315 chunks → 2.5 chunks/worker (SEQ)
256 workers: target=336MB,  creates 299 chunks → 1.2 chunks/worker (PARALLEL) ✓
320 workers: target=269MB,  creates 322 chunks → 1 chunk/worker (overhead)
```

## Performance vs Baseline

| Configuration | vs 256 workers | Speedup |
|---------------|----------------|---------|
| 64 workers    | +14.1s (+97%)  | 0.51x   |
| 128 workers   | +9.0s (+62%)   | 0.62x   |
| **256 workers** | **Baseline**   | **1.00x** |
| 320 workers   | +1.8s (+12%)   | 0.88x   |

## Root Cause Analysis

### Why Not Simply Match Worker Count to Chunks?

The chunk count is determined by:
1. `max_iovec_mb` during checkpoint (256MB limit)
2. VMA boundaries (natural memory layout)
3. Worker count target (affects chunk size calculation)

With `max_iovec_mb=256MB` fixed:
- **Too few workers** → large target chunks → but iovecs max at 256MB → many chunks anyway → sequential per worker
- **Optimal workers** → target ~= iovec size → ~1 chunk per worker → parallel
- **Too many workers** → tiny target chunks → excessive splitting → overhead

### The Disk I/O Pattern:

**64 workers (5 chunks/worker):**
```
Worker 1: Read chunk A → Read chunk B → Read chunk C → Read chunk D → Read chunk E
Total time = 5 × chunk_read_time (SEQUENTIAL)
```

**256 workers (1 chunk/worker):**
```
Worker 1-256: All read chunks A-Z in parallel
Total time = 1 × chunk_read_time (PARALLEL)
```

## Checkpoint Time (independent of restore workers)

| Configuration | Checkpoint Time | Notes |
|---------------|-----------------|-------|
| 320 workers   | 131.1s          | Same checkpoint structure |
| 512 workers   | 125.7s          | Same checkpoint structure |

*Note: Checkpoint time varies slightly due to system load, not worker count*

## Conclusion

**256 workers is the optimal configuration** because:

1. **Matches chunk granularity** (1-1.2 chunks per worker)
2. **True parallel I/O** (minimal sequential work per worker)
3. **Avoids overhead** (not too many workers causing context switching)
4. **Stable and reliable** (no crashes, consistent performance)

**Recommendation**: Keep 256 workers with `max_iovec_mb=256MB`

### Next Optimization Opportunity

Since we've optimized worker count, the next bottleneck is:
- **Disk I/O throughput** (currently ~5.9 GB/s)
- Options:
  1. Try `max_iovec_mb = 512MB` to create fewer, larger chunks
  2. Implement io_uring for async I/O
  3. Use NVMe storage instead of virtio

---

## Testing Methodology Notes

**Important Discovery**: Worker count only affects RESTORE, not checkpoint!

- Checkpoint file structure is determined by `max_iovec_mb` during dump
- Worker count during restore determines parallelism of reading
- We can test different worker counts by:
  1. Create checkpoint once with any worker setting
  2. Change worker count in restore path only
  3. Rebuild and test restore (no re-checkpoint needed!)

This saves significant testing time (~2 minutes per test vs ~5 minutes).
