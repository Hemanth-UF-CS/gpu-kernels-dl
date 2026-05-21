"""
Tiled matrix multiplication in Triton.

Implements the classic blocked GEMM algorithm with configurable tile sizes.
Validates numerically against torch.matmul and benchmarks against it.

Reference: Triton GEMM tutorial
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Tiled GEMM: C = A @ B
    Each program computes a (BLOCK_M x BLOCK_N) output tile.
    Tiles are grouped in L2-cache-friendly swizzled order.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Pointers to this tile's row / col
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 128,
    block_n: int = 256,
    block_k: int = 64,
    group_size_m: int = 8,
) -> torch.Tensor:
    """
    Compute C = A @ B using the Triton tiled kernel.

    Args:
        a: (M, K) float16 tensor on CUDA
        b: (K, N) float16 tensor on CUDA
    Returns:
        c: (M, N) float16 tensor on CUDA
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.shape[1] == b.shape[0], f"Shape mismatch: {a.shape} @ {b.shape}"

    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
    )
    return c


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(M: int = 512, N: int = 512, K: int = 512, atol: float = 1e-2) -> bool:
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")

    ref = torch.matmul(a, b)
    out = triton_matmul(a, b)

    ok = torch.allclose(ref, out, atol=atol, rtol=1e-2)
    print(f"matmul validation ({M}x{K} @ {K}x{N}): {'PASS' if ok else 'FAIL'}")
    if not ok:
        max_diff = (ref - out).abs().max().item()
        print(f"  max_diff = {max_diff:.4f}")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skipping matmul validation")
    else:
        for size in [256, 512, 1024, 4096]:
            validate(size, size, size)
