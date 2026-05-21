# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [1.2.0] — 2026-05-21

### Added
- Unit tests for all three Triton kernels in `tests/test_kernels.py`;
  CUDA tests auto-skip when no GPU is available
- `benchmark_results.txt` — measured latency / TFLOPS / bandwidth on RTX 3090;
  Triton matmul is ~2× slower than cuBLAS; README explains why
- README: replaced speculative tables with measured numbers; added explanation
  of compute- vs bandwidth-bound bottlenecks and the cuBLAS gap

## [1.1.0] — 2026-05-20

### Fixed
- **`triton_kernels/reduction.py`**: partial sums were allocated with the input
  dtype (`torch.empty(..., dtype=x.dtype)`). For `float16` inputs with large N,
  the accumulation overflows (`float16` max ≈ 65504), silently returning a wrong
  result. Partials are now always `float32` regardless of input dtype; the final
  scalar is cast back to the original dtype. Added `validate_fp16()` to confirm
  the fix.

## [1.0.0] — 2026-05-19

### Added
- `triton_kernels/matmul.py` — tiled GEMM with L2-swizzled program ordering
- `triton_kernels/softmax.py` — numerically-stable fused row-wise softmax
- `triton_kernels/reduction.py` — two-phase parallel sum/max/mean
- `cuda_kernels/matmul.cu` — shared-memory tiled GEMM vs naive; TFLOPS benchmark
- `cuda_kernels/softmax.cu` — 3-pass naive vs 1-pass warp-shuffle fused softmax
- `benchmarks/bench_matmul.py` / `bench_softmax.py` — latency, TFLOPS/bandwidth
  vs PyTorch baselines with CUDA event timing
- `.gitignore` — excludes compiled CUDA binaries, Triton JIT cache, venv
