"""
Benchmark: custom Triton matmul vs torch.matmul across matrix sizes.

Reports latency (ms), TFLOPS, and speedup.  Uses CUDA events for precise timing.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triton_kernels.matmul import triton_matmul
from benchmarks import record

WARMUP = 5
REPS = 50


def tflops(M: int, N: int, K: int, ms: float) -> float:
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def time_fn(fn, warmup: int = WARMUP, reps: int = REPS) -> float:
    """Returns average latency in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def run_benchmark(sizes: list[int]) -> list[dict]:
    print(f"{'Size':>6}  {'torch ms':>10}  {'triton ms':>10}  "
          f"{'torch TFLOPS':>14}  {'triton TFLOPS':>14}  {'speedup':>8}")
    print("-" * 72)

    rows = []
    for size in sizes:
        M = N = K = size
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        torch_ms = time_fn(lambda: torch.matmul(a, b))
        triton_ms = time_fn(lambda: triton_matmul(a, b))

        torch_tf = tflops(M, N, K, torch_ms)
        triton_tf = tflops(M, N, K, triton_ms)
        speedup = torch_ms / triton_ms

        print(f"{size:>6}  {torch_ms:>10.3f}  {triton_ms:>10.3f}  "
              f"{torch_tf:>14.2f}  {triton_tf:>14.2f}  {speedup:>8.2f}x")
        rows.append({"size": size, "torch_ms": torch_ms, "triton_ms": triton_ms,
                     "torch_tflops": torch_tf, "triton_tflops": triton_tf})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/matmul.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available — cannot benchmark GPU kernels")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}  Triton {triton.__version__}\n")

    sizes = [256, 512, 1024, 2048, 4096]
    record.write(args.out, run_benchmark(sizes))


if __name__ == "__main__":
    main()
