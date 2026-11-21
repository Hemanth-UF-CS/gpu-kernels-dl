# gpu-kernels-dl

Matrix multiply, row-wise softmax and parallel reduction written as GPU kernels in Triton and
CUDA C++, each checked against PyTorch or a CPU reference and benchmarked with CUDA events.

## Results (RTX 3090)

torch 2.5.1+cu121, Triton 3.1.0, CUDA 12.1. Raw numbers are in
[`results/`](results/), written by the benchmark scripts.

**Matmul, fp16 in, fp32 accumulate (TFLOPS, higher is better)**

| N (N x N x N) | cuBLAS (`torch.matmul`) | Triton | Triton / cuBLAS |
|---:|---:|---:|---:|
| 256 | 4.8 | 1.0 | 0.21x |
| 512 | 32.7 | 7.9 | 0.24x |
| 1024 | 55.4 | 54.3 | 0.98x |
| 2048 | 67.6 | 64.3 | 0.95x |
| 4096 | 66.3 | 69.3 | 1.05x |

From N = 1024 up, the Triton kernel matches cuBLAS (0.95x to 1.05x). Below that the problem
is a few microseconds of GPU work, and the Python-side launch of a Triton kernel (about 30 us)
dominates.

**Softmax, fp32 (effective GB/s, one read and one write of the tensor)**

| Shape | `torch.softmax` | Triton | Triton / PyTorch |
|---:|---:|---:|---:|
| 512 x 512 | 491 | 130 | 0.26x |
| 1024 x 1024 | 690 | 513 | 0.74x |
| 2048 x 2048 | 580 | 806 | 1.39x |
| 4096 x 4096 | 825 | 833 | 1.01x |
| 8192 x 4096 | 825 | 838 | 1.02x |

Softmax is bandwidth-bound. For rows of 2048 columns and wider, the Triton kernel reaches
about 830 GB/s, close to the card's 936 GB/s, and matches or beats PyTorch. Small tensors are
again dominated by launch overhead.

## Kernels

| Kernel | Triton | CUDA C++ | Technique |
|---|---|---|---|
| Matmul | `triton_kernels/matmul.py` | `cuda_kernels/matmul.cu` | Blocked GEMM with `tl.dot` (tensor cores), L2-friendly grouped launch order, autotuned tile, pipeline and warp configs. The CUDA version is a 32x32 shared-memory tiled fp32 kernel next to a naive one. |
| Softmax | `triton_kernels/softmax.py` | `cuda_kernels/softmax.cu` | One program per row. Triton keeps the row in registers (one read, one write). CUDA uses block-wide max and sum reductions with warp shuffles. |
| Reduction | `triton_kernels/reduction.py` | | Two-phase sum, max and mean. Partial results are always fp32, so fp16 inputs keep precision. |

## Run

```bash
pip install -r requirements.txt
pytest -q                                        # 20 tests, skipped without a GPU

python benchmarks/bench_matmul.py --out results/matmul.json
python benchmarks/bench_softmax.py --out results/softmax.json

cd cuda_kernels && make ARCH=sm_86               # sm_89 for RTX 40xx, sm_80 for A100
./matmul 333 517 129                             # checked against a CPU reference
./softmax 4096 4096                              # checked against a CPU reference
```

## Notes

- **Tile configuration decides matmul speed.** A fixed 128x256x64 tile with the default 4
  warps reached only 10.6 TFLOPS at N = 4096 (0.16x cuBLAS), because each thread held too
  large an accumulator and spilled registers. `triton.autotune` over 8 standard configs, with
  4 or 8 warps and 3 to 5 pipeline stages, brings it to 69.3 TFLOPS.
- **Block-wide reductions need two levels.** A warp shuffle reduces only 32 lanes. With 256
  threads per row, both CUDA softmax kernels first kept one warp's partial result and dropped
  the rest, so they disagreed with each other. They now reduce each warp, then reduce the
  per-warp results, and match a double-precision CPU reference within 1e-8.
- **The CUDA C++ matmul is a teaching kernel.** It uses fp32 on CUDA cores without tensor cores
  (2.85 TFLOPS at N = 4096, against 2.14 for the naive kernel). The gap to cuBLAS is what
  tensor cores, register tiling and async copies buy. The Triton kernel gets them through
  `tl.dot` and software pipelining.

## References

- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [CUDA C++ Programming Guide: shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
