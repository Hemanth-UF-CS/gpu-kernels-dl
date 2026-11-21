# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed
- `results/rtx3090_20260521.txt` did not match what the benchmarks produce on an RTX 3090
  (e.g. cuBLAS reaches 67 TFLOPS at N = 4096, not 40). Removed it. The benchmarks now write
  JSON with GPU and version metadata, and the README tables come from that output.
- Triton matmul: autotune tile, pipeline and warp configs. At N = 4096 it went from
  10.6 to 69.3 TFLOPS on an RTX 3090.
- CUDA softmax: both kernels reduced only within one warp of a 256-thread block. They now use
  block-wide reductions and are checked against a CPU reference.
- `cuda_kernels/matmul.cu` did not compile (missing `<functional>`). `ARCH` can now be set
  on the make command line.
- `test_sum_fp16_no_overflow` expected a sum (524,288) that fp16 cannot represent. It now
  checks fp32 accumulation with a sum fp16 can hold.

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
