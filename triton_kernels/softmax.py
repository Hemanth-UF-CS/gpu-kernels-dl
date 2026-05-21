"""
Row-wise softmax in Triton.

Each Triton program handles one row: load row → subtract max (for numerical
stability) → exponentiate → divide by sum.  This fuses all four passes into a
single kernel, avoiding the O(n) intermediate stores that a naive implementation
requires.

Validates numerical correctness against torch.softmax and benchmarks both.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program = one row.
    BLOCK_SIZE must be >= n_cols (pad the load with -inf).
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load row; pad out-of-bound lanes with -inf so they don't affect max/sum
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))

    # Numerically-stable softmax: subtract max before exp
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise softmax of a 2-D tensor using the Triton kernel.

    Args:
        x: (n_rows, n_cols) float32 or float16 tensor on CUDA
    Returns:
        y: same shape and dtype, softmax applied along dim=1
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.ndim == 2, "Expected 2D input"

    n_rows, n_cols = x.shape
    # BLOCK_SIZE must be a power of 2 >= n_cols
    block_size = triton.next_power_of_2(n_cols)
    # Clamp to what fits in shared memory (heuristic: 128K elements)
    block_size = min(block_size, 131072)

    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16

    y = torch.empty_like(x)

    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=block_size,
    )
    return y


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(n_rows: int = 1024, n_cols: int = 1024, atol: float = 1e-5) -> bool:
    torch.manual_seed(0)
    x = torch.randn(n_rows, n_cols, dtype=torch.float32, device="cuda")

    ref = torch.softmax(x, dim=1)
    out = triton_softmax(x)

    ok = torch.allclose(ref, out, atol=atol, rtol=1e-4)
    print(f"softmax validation ({n_rows}x{n_cols}): {'PASS' if ok else 'FAIL'}")
    if not ok:
        max_diff = (ref - out).abs().max().item()
        print(f"  max_diff = {max_diff:.6f}")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skipping softmax validation")
    else:
        for shape in [(512, 512), (1024, 1024), (4096, 4096), (8192, 8192)]:
            validate(*shape)
