# Custom GPU Kernels for Deep Learning Primitives

Custom GPU kernels for tiled matrix multiplication, row-wise softmax, and parallel
reduction — implemented in both **Triton** and **CUDA C++**, validated against PyTorch
baselines, and benchmarked with CUDA events.

Built to understand what vendor-optimized libraries actually do that a clean
educational kernel does not.

## Kernels

| Kernel | Triton | CUDA C++ | What it demonstrates |
|--------|--------|----------|----------------------|
| Matrix multiply | `triton_kernels/matmul.py` | `cuda_kernels/matmul.cu` | Shared-memory tiling, L2-swizzled program order, `tl.dot` vs naive loop |
| Row-wise softmax | `triton_kernels/softmax.py` | `cuda_kernels/softmax.cu` | Numerically-stable fused softmax, warp `__shfl_xor_sync` reductions |
| Reduction (sum/max/mean) | `triton_kernels/reduction.py` | — | Two-phase parallel reduction, shared-memory tree reduce |

## Quick start

### Triton kernels

```bash
pip install -r requirements.txt

# Validate correctness
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
# Edit ARCH= in Makefile to match your GPU (sm_80=A100, sm_86=RTX 30xx, sm_89=RTX 40xx)
make

./matmul 1024 1024 1024
./softmax 4096 4096
```

## Benchmark results

Measured on **RTX 3090** (24 GB, 35.6 TFLOPS fp16 TC, 936 GB/s), PyTorch 2.3.0,
Triton 2.3.0.  50 iterations after 5 warmup, CUDA event timing.

### Matrix multiplication — fp16, TFLOPS

| Size   | torch.matmul | Triton kernel | Triton / cuBLAS |
|--------|-------------|---------------|-----------------|
| 256    | 0.73        | 0.40          | 0.55×           |
| 512    | 5.18        | 2.26          | 0.44×           |
| 1024   | 19.16       | 8.65          | 0.45×           |
| 2048   | 34.16       | 17.76         | 0.52×           |
| 4096   | 39.93       | 20.19         | 0.51×           |

**The Triton kernel is ~2× slower than cuBLAS and that is the correct result.**

cuBLAS closes the gap through techniques this kernel does not implement:
- **Async double-buffering** (`cp.async` / `ldmatrix`): overlaps the global→shared load
  of tile `t+1` with the matrix-multiply of tile `t`. Our kernel stalls on each load.
- **`mma.sync` warp-level instructions**: cuBLAS emits Tensor Core ops directly via PTX;
  `tl.dot` maps to Tensor Cores too, but with coarser layout control.
- **Epilogue fusion**: cuBLAS fuses bias add, ReLU, etc. into the last write; our kernel
  writes `C` in a plain store.

The TFLOPS gap narrows at large sizes (39.93 vs 20.19 at 4096) because arithmetic
intensity grows with matrix size and the pipeline stalls become proportionally cheaper.

### Row-wise softmax — fp32, effective memory bandwidth (GB/s)

| Shape        | torch.softmax | Triton kernel | Triton / PyTorch |
|--------------|---------------|---------------|------------------|
| (512, 512)   | 54            | 33            | 0.61×            |
| (1024, 1024) | 163           | 122           | 0.75×            |
| (2048, 2048) | 470           | 388           | 0.83×            |
| (4096, 4096) | 452           | 393           | 0.87×            |
| (8192, 4096) | 460           | 406           | 0.88×            |

Softmax is memory-bandwidth bound; both implementations approach the hardware limit
(~500 GB/s effective) at large sizes.  The gap at small shapes (512×512) is kernel-launch
overhead: Triton's JIT compilation and runtime dispatch add fixed cost that dominates
when the actual work is tiny.  At 8192×4096 the kernel is within 12% of PyTorch's
cuDNN-backed implementation.

## Design notes

### Tiled matrix multiplication

Naive GEMM reads each row of A and column of B K times from global memory.  Tiled GEMM
loads 32×32 sub-tiles into shared memory (4–10 TB/s effective) and reuses them TILE_SIZE
times, reducing global traffic by `TILE_SIZE`:

```
naive   global reads: 2 × M × N × K bytes
tiled   global reads: M × N × K × (2 / TILE_SIZE) bytes
```

The Triton kernel adds L2-swizzled launch order (`GROUP_SIZE_M`) so that consecutive
programs access overlapping B tiles and improve L2 hit rate.

### Fused softmax

Three passes (find max → compute exp-sum → normalize) require three global round-trips.
The fused kernel performs a single pass: thread block finds its local max via warp shuffle
(`__shfl_xor_sync`), reduces across the block via shared memory, then computes exp and
sum in a second sweep of the same data that is still in registers.  Result: one read +
one write instead of three reads + two writes.

### Parallel reduction

Two-phase design: phase 1 reduces BLOCK_SIZE-element tiles to per-block partial sums;
phase 2 reduces the (much smaller) partial array.  For arrays fitting in one block,
phase 2 is skipped.  Partials are always accumulated in float32 regardless of input
dtype to prevent overflow on large float16 arrays.

## What I learned

- **Compute- vs bandwidth-bound**: matmul is compute-bound (arithmetic intensity ≫ 1),
  softmax is bandwidth-bound (one exp per byte). Different bottlenecks need different
  optimization strategies.
- **Occupancy vs register pressure**: wider tiles improve arithmetic intensity but
  increase register use and reduce SM occupancy. The 128×256×64 tile config was chosen
  after profiling SM utilization with Nsight Compute.
- **Warp divergence**: kept to zero by uniform control flow and aligned tile boundaries
  with explicit masking rather than `if` branches.
- **dtype tradeoffs**: fp16 halves memory traffic and hits Tensor Cores; accumulating in
  fp32 avoids precision loss in the reduction tree.
- **What cuBLAS actually does**: reading the CUTLASS source and PTX disassembly made the
  gap between "correct" and "fast" concrete.

## References

- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://arxiv.org/abs/1912.10209)
- [CUDA Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)
- [Nsight Compute — Roofline Analysis](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
