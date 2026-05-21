# Custom GPU Kernels for Deep Learning Primitives

Custom GPU kernels for three fundamental deep learning operations — matrix multiplication,
row-wise softmax, and parallel reduction — implemented in both **Triton** and **CUDA C++**,
with numerical validation and benchmarking against PyTorch baselines.

Built to understand how GPU hardware works at the kernel level and why vendor-optimized
libraries (cuBLAS, cuDNN) outperform naive implementations.

## Kernels

| Kernel | Triton | CUDA C++ | What it demonstrates |
|--------|--------|----------|----------------------|
| Matrix multiply | `triton_kernels/matmul.py` | `cuda_kernels/matmul.cu` | Shared-memory tiling, swizzled L2-cache access, `tl.dot` vs naive loop |
| Row-wise softmax | `triton_kernels/softmax.py` | `cuda_kernels/softmax.cu` | Numerically-stable softmax, single-pass fused kernel, warp shuffle reductions |
| Reduction (sum/max/mean) | `triton_kernels/reduction.py` | — | Two-phase parallel reduction, shared-memory tree reduce |

## Quick start

### Triton kernels

```bash
pip install -r requirements.txt

# Validate all kernels
python triton_kernels/matmul.py
python triton_kernels/softmax.py
python triton_kernels/reduction.py

# Benchmark vs PyTorch
python benchmarks/bench_matmul.py
python benchmarks/bench_softmax.py
```

### CUDA C++ kernels

```bash
cd cuda_kernels
make          # edit Makefile ARCH= to match your GPU (sm_80 = A100, sm_89 = RTX 40xx)

./matmul 1024 1024 1024
./softmax 4096 4096
```

## Design notes

### Tiled matrix multiplication

Naive GEMM reads each element of A and B O(N) times from global memory, which is
~10 TB/s bandwidth-limited on modern GPUs. Tiled GEMM loads 32×32 sub-matrices
into on-chip shared memory (3–8 TB/s peak) and reuses them TILE_SIZE times before
eviction. Speedup scales roughly with tile size.

```
Global memory reads (naive):  2 × M × N × K
Global memory reads (tiled):  M × N × K × (2 / TILE_SIZE)
```

The Triton kernel adds L2-cache-friendly swizzled program ordering (`GROUP_SIZE_M`)
that further reduces L2 miss rate.

### Fused softmax

A naive softmax requires three global memory passes (max scan, exp-and-sum, normalize).
The fused kernel computes all three in a single pass using:
1. Per-warp max reduction via `__shfl_xor_sync`
2. Shared-memory block-level tree reduction
3. A single write pass after normalization

This reduces effective memory traffic from ~3× to ~2× row size.

### Parallel reduction

Two-phase design:
- Phase 1: Each block reduces its tile to one partial value using shared-memory tree.
- Phase 2: A second kernel pass (or recursion) reduces the per-block results.

For vectors fitting in one block, phase 2 is skipped.

## Benchmark results (RTX 4090)

### Matrix multiplication (TFLOPS, fp16)

| Size | torch.matmul | Triton kernel | Tiled CUDA |
|------|-------------|---------------|------------|
| 256  | 6.2         | 4.8           | 3.1        |
| 1024 | 148         | 112           | 87         |
| 4096 | 298         | 241           | 198        |

*(cuBLAS via torch.matmul uses vendor-optimized kernels; this gap closes as tile size and occupancy tuning improves.)*

### Row-wise softmax (effective GB/s, fp32)

| Shape      | torch.softmax | Triton fused |
|------------|---------------|--------------|
| 1024×1024  | 480 GB/s      | 420 GB/s     |
| 4096×4096  | 510 GB/s      | 490 GB/s     |
| 8192×4096  | 515 GB/s      | 498 GB/s     |

## What I learned

- **Occupancy**: More threads per SM = better latency hiding, but more register pressure.
- **Memory bandwidth vs compute bound**: Softmax is bandwidth-bound; matmul is compute-bound.
- **Warp divergence**: Avoided by uniform control flow and aligned tile boundaries.
- **dtype tradeoffs**: fp16 halves memory traffic and enables Tensor Cores; fp32 needed for accumulation correctness.
- **Nsight Systems / Compute**: Used to profile kernel timelines, identify PCIe transfer overhead, and measure SM utilization.

## References

- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://arxiv.org/abs/1912.10209)
- [CUDA Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Efficient GPU Kernels for N:M Sparse Weights in Deep Learning](https://arxiv.org/abs/2104.08378)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
