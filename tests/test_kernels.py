"""
Unit tests for Triton kernels.

Tests that require CUDA are skipped automatically when no GPU is available.
CPU-only tests cover the Python wrapper logic (shapes, types, error handling).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Imports (deferred so CPU-only runs don't fail on triton import itself)
# ---------------------------------------------------------------------------
def _import_kernels():
    from triton_kernels.matmul import triton_matmul, validate as val_matmul
    from triton_kernels.softmax import triton_softmax, validate as val_softmax
    from triton_kernels.reduction import triton_sum, triton_max, triton_mean, validate as val_reduce
    return triton_matmul, val_matmul, triton_softmax, val_softmax, triton_sum, triton_max, triton_mean, val_reduce


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------
class TestMatmul:
    @CUDA
    def test_output_shape(self):
        triton_matmul, *_ = _import_kernels()
        a = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        b = torch.randn(128, 32, dtype=torch.float16, device="cuda")
        c = triton_matmul(a, b)
        assert c.shape == (64, 32)

    @CUDA
    def test_output_dtype(self):
        triton_matmul, *_ = _import_kernels()
        a = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        b = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        c = triton_matmul(a, b)
        assert c.dtype == torch.float16

    @CUDA
    def test_numerical_correctness_small(self):
        triton_matmul, *_ = _import_kernels()
        torch.manual_seed(0)
        a = torch.randn(128, 128, dtype=torch.float16, device="cuda")
        b = torch.randn(128, 128, dtype=torch.float16, device="cuda")
        ref = torch.matmul(a, b)
        out = triton_matmul(a, b)
        assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2), \
            f"max diff: {(ref - out).abs().max().item():.4f}"

    @CUDA
    def test_non_square(self):
        triton_matmul, *_ = _import_kernels()
        torch.manual_seed(1)
        a = torch.randn(256, 512, dtype=torch.float16, device="cuda")
        b = torch.randn(512, 128, dtype=torch.float16, device="cuda")
        ref = torch.matmul(a, b)
        out = triton_matmul(a, b)
        assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2)

    @CUDA
    @pytest.mark.parametrize("size", [256, 512, 1024])
    def test_validate_passes(self, size):
        _, val_matmul, *_ = _import_kernels()
        assert val_matmul(size, size, size)

    def test_raises_on_cpu_input(self):
        triton_matmul, *_ = _import_kernels()
        a = torch.randn(64, 64, dtype=torch.float16)
        b = torch.randn(64, 64, dtype=torch.float16)
        with pytest.raises(AssertionError):
            triton_matmul(a, b)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
class TestSoftmax:
    @CUDA
    def test_output_shape(self):
        _, _, triton_softmax, *_ = _import_kernels()
        x = torch.randn(32, 128, device="cuda")
        y = triton_softmax(x)
        assert y.shape == x.shape

    @CUDA
    def test_rows_sum_to_one(self):
        _, _, triton_softmax, *_ = _import_kernels()
        torch.manual_seed(0)
        x = torch.randn(64, 256, device="cuda")
        y = triton_softmax(x)
        row_sums = y.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(64, device="cuda"), atol=1e-5)

    @CUDA
    def test_numerical_correctness(self):
        _, _, triton_softmax, *_ = _import_kernels()
        torch.manual_seed(0)
        x = torch.randn(128, 512, device="cuda")
        ref = torch.softmax(x, dim=1)
        out = triton_softmax(x)
        assert torch.allclose(ref, out, atol=1e-5, rtol=1e-4)

    @CUDA
    def test_stable_with_large_values(self):
        """Numerically-stable: large inputs shouldn't produce NaN/Inf."""
        _, _, triton_softmax, *_ = _import_kernels()
        x = torch.full((8, 256), 1e4, device="cuda")
        y = triton_softmax(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_raises_on_1d_input(self):
        _, _, triton_softmax, *_ = _import_kernels()
        if torch.cuda.is_available():
            x = torch.randn(64, device="cuda")
            with pytest.raises(AssertionError):
                triton_softmax(x)


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
class TestReduction:
    @CUDA
    def test_sum_correctness(self):
        *_, triton_sum, triton_max, triton_mean, _ = _import_kernels()
        torch.manual_seed(0)
        x = torch.randn(1 << 20, device="cuda")
        assert torch.allclose(triton_sum(x), x.sum(), rtol=1e-3, atol=1e-3)

    @CUDA
    def test_max_correctness(self):
        *_, triton_sum, triton_max, triton_mean, _ = _import_kernels()
        torch.manual_seed(0)
        x = torch.randn(1 << 20, device="cuda")
        assert torch.allclose(triton_max(x), x.max(), rtol=1e-5, atol=1e-5)

    @CUDA
    def test_mean_correctness(self):
        *_, triton_sum, triton_max, triton_mean, _ = _import_kernels()
        torch.manual_seed(0)
        x = torch.randn(1 << 16, device="cuda")
        assert torch.allclose(triton_mean(x), x.mean(), rtol=1e-3, atol=1e-3)

    @CUDA
    def test_sum_fp16_no_overflow(self):
        """fp16 input is accumulated in fp32, so precision is kept for large N.

        65,536 x 0.5 = 32,768 is exactly representable in fp16, but an fp16 running sum
        stalls once the spacing between fp16 values exceeds 0.5 (around 2,048).
        """
        *_, triton_sum, triton_max, triton_mean, _ = _import_kernels()
        n = 1 << 16
        x = torch.full((n,), 0.5, device="cuda", dtype=torch.float16)
        result = triton_sum(x)
        assert result.dtype == torch.float16
        assert result.item() == 0.5 * n

    @CUDA
    @pytest.mark.parametrize("n", [1024, 1 << 16, 1 << 20])
    def test_validate_passes(self, n):
        *_, val_reduce = _import_kernels()
        assert val_reduce(n)
