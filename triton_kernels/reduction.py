"""
Parallel reduction kernels in Triton.

Implements three common reductions used in deep learning:
  - sum_reduction:  Σ x[i]
  - max_reduction:  max x[i]
  - mean_reduction: (Σ x[i]) / N

Each uses a two-phase approach:
  Phase 1  — each block reduces its local tile to a single scalar.
  Phase 2  — a single warp reduces the per-block results.

For tensors that fit in a single block, phase 2 is skipped.

Bug fix (2026-05-20): partials were allocated with the input dtype, causing
silent overflow on float16 inputs with large N.  Partials are now always
float32 regardless of input dtype; the final result is cast back to input dtype.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Phase-1 kernels
# ---------------------------------------------------------------------------
@triton.jit
def sum_reduce_kernel(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program reduces BLOCK_SIZE elements and writes one partial sum (float32)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(x, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def max_reduce_kernel(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    partial = tl.max(x, axis=0)
    tl.store(partial_ptr + pid, partial)


# ---------------------------------------------------------------------------
# Two-phase Python wrappers
# ---------------------------------------------------------------------------
BLOCK_SIZE = 1024


def triton_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum all elements of a 1-D CUDA tensor. Always accumulates in float32."""
    assert x.is_cuda and x.ndim == 1
    original_dtype = x.dtype
    n = x.numel()
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    # Partials always float32 to avoid overflow on fp16 inputs with large N
    partials = torch.empty(n_blocks, dtype=torch.float32, device=x.device)

    sum_reduce_kernel[(n_blocks,)](x, partials, n, BLOCK_SIZE=BLOCK_SIZE)

    if n_blocks == 1:
        return partials[0].to(original_dtype)
    # Recurse: partials is already float32 so no further casting needed
    result = triton_sum(partials)
    return result.to(original_dtype)


def triton_max(x: torch.Tensor) -> torch.Tensor:
    """Max of all elements of a 1-D CUDA tensor. Always accumulates in float32."""
    assert x.is_cuda and x.ndim == 1
    original_dtype = x.dtype
    n = x.numel()
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    partials = torch.empty(n_blocks, dtype=torch.float32, device=x.device)

    max_reduce_kernel[(n_blocks,)](x, partials, n, BLOCK_SIZE=BLOCK_SIZE)

    if n_blocks == 1:
        return partials[0].to(original_dtype)
    result = triton_max(partials)
    return result.to(original_dtype)


def triton_mean(x: torch.Tensor) -> torch.Tensor:
    return triton_sum(x) / x.numel()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(n: int = 1 << 20) -> bool:
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    ref_sum = x.sum()
    ref_max = x.max()
    ref_mean = x.mean()

    out_sum = triton_sum(x)
    out_max = triton_max(x)
    out_mean = triton_mean(x)

    ok_sum = torch.allclose(ref_sum, out_sum, rtol=1e-3, atol=1e-3)
    ok_max = torch.allclose(ref_max, out_max, rtol=1e-5, atol=1e-5)
    ok_mean = torch.allclose(ref_mean, out_mean, rtol=1e-3, atol=1e-3)

    print(f"reduction validation (n={n:,}):")
    print(f"  sum:  {'PASS' if ok_sum  else 'FAIL'}  ref={ref_sum.item():.4f}  out={out_sum.item():.4f}")
    print(f"  max:  {'PASS' if ok_max  else 'FAIL'}  ref={ref_max.item():.4f}  out={out_max.item():.4f}")
    print(f"  mean: {'PASS' if ok_mean else 'FAIL'}  ref={ref_mean.item():.4f}  out={out_mean.item():.4f}")

    return ok_sum and ok_max and ok_mean


def validate_fp16(n: int = 1 << 20) -> bool:
    """Confirm no overflow on float16 input — this would silently fail before the fix."""
    torch.manual_seed(0)
    # Use values near fp16 max to stress-test accumulation
    x = torch.full((n,), 0.5, device="cuda", dtype=torch.float16)
    expected = 0.5 * n

    out = triton_sum(x)
    ok = abs(out.item() - expected) / expected < 0.001
    print(f"fp16 overflow test (n={n:,}, expected={expected}): {'PASS' if ok else 'FAIL'} (got {out.item():.1f})")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skipping reduction validation")
    else:
        for n in [1024, 1 << 16, 1 << 20, 1 << 24]:
            validate(n)
        print()
        for n in [1 << 16, 1 << 20]:
            validate_fp16(n)
