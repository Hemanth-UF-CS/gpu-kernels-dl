"""
Benchmark: custom Triton row-wise softmax vs torch.softmax across sizes.

Reports latency (ms), effective memory bandwidth (GB/s), and speedup.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triton_kernels.softmax import triton_softmax
from benchmarks import record

WARMUP = 5
REPS = 100


def bandwidth_gbs(rows: int, cols: int, ms: float) -> float:
    bytes_rw = 2.0 * rows * cols * 4  # read + write, float32
    return bytes_rw / (ms * 1e-3) / 1e9


def time_fn(fn, warmup: int = WARMUP, reps: int = REPS) -> float:
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


def run_benchmark(shapes: list[tuple[int, int]]) -> list[dict]:
    print(f"{'Shape':>14}  {'torch ms':>10}  {'triton ms':>10}  "
          f"{'torch GB/s':>12}  {'triton GB/s':>12}  {'speedup':>8}")
    print("-" * 72)

    out = []
    for rows, cols in shapes:
        x = torch.randn(rows, cols, dtype=torch.float32, device="cuda")

        torch_ms = time_fn(lambda: torch.softmax(x, dim=1))
        triton_ms = time_fn(lambda: triton_softmax(x))

        torch_bw = bandwidth_gbs(rows, cols, torch_ms)
        triton_bw = bandwidth_gbs(rows, cols, triton_ms)
        speedup = torch_ms / triton_ms

        print(f"{str((rows, cols)):>14}  {torch_ms:>10.3f}  {triton_ms:>10.3f}  "
              f"{torch_bw:>12.1f}  {triton_bw:>12.1f}  {speedup:>8.2f}x")
        out.append({"rows": rows, "cols": cols, "torch_ms": torch_ms, "triton_ms": triton_ms,
                    "torch_gbs": torch_bw, "triton_gbs": triton_bw})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/softmax.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available — cannot benchmark GPU kernels")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}  Triton {triton.__version__}\n")

    shapes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 4096),
    ]
    record.write(args.out, run_benchmark(shapes))


if __name__ == "__main__":
    main()
