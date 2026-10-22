import numpy as np
import pytest
from conftest import RNG

from aljabr.concrete import (
    DFT,
    CircConv,
    Conv,
    Diag,
    Diff,
    Identity,
    RealDFT,
    Sampling,
    Slice,
)
from aljabr.utils import dottest, fwadjtest

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_shape(self):
        op = Identity((4, 4))
        assert op.ishape == op.oshape == (4, 4)

    def test_forward_is_passthrough(self):
        op = Identity((3,))
        x = RNG.standard_normal(3)
        np.testing.assert_array_equal(op.forward(x), x)

    def test_asmatrix_is_eye(self):
        np.testing.assert_array_equal(Identity((3,)).asmatrix(), np.eye(3))

    def test_asmatrix_like_dtype(self):
        mat = Identity((3,)).asmatrix(like=np.zeros(1, dtype=np.float32))
        assert mat.dtype == np.float32

    def test_dottest(self):
        assert dottest(Identity((4,)), num=5)


# ---------------------------------------------------------------------------
# Diag
# ---------------------------------------------------------------------------


class TestDiag:
    def test_shape(self):
        op = Diag(RNG.standard_normal(5))
        assert op.ishape == op.oshape == (5,)

    def test_forward(self):
        d = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(Diag(d).forward(np.ones(3)), d)

    def test_adjoint_conjugate(self):
        d = np.array([1 + 1j, 2 - 1j])
        np.testing.assert_array_equal(
            Diag(d).adjoint(np.ones(2, dtype=complex)), d.conj()
        )

    def test_fwadj(self):
        d = RNG.standard_normal(5)
        x = RNG.standard_normal(5)
        np.testing.assert_allclose(Diag(d).fwadj(x), d**2 * x)

    def test_dottest(self):
        assert dottest(Diag(RNG.standard_normal(5)), num=5)


# ---------------------------------------------------------------------------
# DFT
# ---------------------------------------------------------------------------


class TestDFT:
    def test_shape(self):
        op = DFT((8,), 1)
        assert op.ishape == op.oshape == (8,)

    def test_unitary_fwadj(self):
        assert fwadjtest(DFT((8,), 1), num=5)

    def test_dottest_1d(self):
        assert dottest(DFT((8,), 1), num=5)

    def test_dottest_2d(self):
        assert dottest(DFT((4, 4), 2), num=3)

    def test_asmatrix_is_complex_unitary(self):
        mat = DFT((4, 4), 2).asmatrix()
        assert np.iscomplexobj(mat)
        np.testing.assert_allclose(mat.conj().T @ mat, np.eye(16), atol=1e-12)

    def test_asmatrix_2d_is_kron(self):
        f1 = np.fft.fft(np.eye(4), axis=0, norm="ortho")
        np.testing.assert_allclose(DFT((4, 4), 2).asmatrix(), np.kron(f1, f1))

    def test_asmatrix_adjoint(self):
        op = DFT((8,), 1)
        np.testing.assert_allclose(op.H.asmatrix(), op.asmatrix().conj().T)


# ---------------------------------------------------------------------------
# RealDFT
# ---------------------------------------------------------------------------


class TestRealDFT:
    def test_output_shape(self):
        op = RealDFT((8,), 1)
        assert op.ishape == (8,)
        assert op.oshape == (5,)  # 8//2 + 1

    def test_fwadjtest(self):
        assert fwadjtest(RealDFT((8,), 1), num=5)

    def test_asmatrix_is_complex(self):
        mat = RealDFT((8,), 1).asmatrix()
        assert mat.shape == (5, 8)
        np.testing.assert_allclose(mat, np.fft.rfft(np.eye(8), axis=0, norm="ortho"))

    # The adjoint is the one of DFT on the Hermitian-symmetric spectrum, that is
    # for the weighted inner product of the half spectrum (see docstring).

    @pytest.mark.parametrize(
        ("shape", "ndim"), [((8,), 1), ((9,), 1), ((6, 8), 2), ((3, 5, 7), 2)]
    )
    def test_weighted_dottest(self, shape, ndim):
        op = RealDFT(shape, ndim)
        x = RNG.standard_normal(op.ishape)
        u = RNG.standard_normal(op.oshape) + 1j * RNG.standard_normal(op.oshape)
        weights = half_spectrum_weights(shape[-1])
        np.testing.assert_allclose(
            np.vdot(op.adjoint(u), x),
            np.real(np.sum(weights * np.conj(u) * op.forward(x))),
        )

    @pytest.mark.parametrize("size", [8, 9])
    def test_weighted_parseval(self, size):
        x = RNG.standard_normal(size)
        coeffs = RealDFT((size,), 1).forward(x)
        np.testing.assert_allclose(
            np.sum(half_spectrum_weights(size) * np.abs(coeffs) ** 2), np.sum(x**2)
        )

    def test_not_standard_adjoint(self):
        assert not dottest(RealDFT((8,), 1), num=3)


def half_spectrum_weights(size: int) -> np.ndarray:
    """Weights 1 for frequencies 0 and size/2 (if even), 2 for the others."""
    weights = np.full(size // 2 + 1, 2.0)
    weights[0] = 1
    if size % 2 == 0:
        weights[-1] = 1
    return weights


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_shape_1d(self):
        op = Diff(axis=0, ishape=(5,))
        assert op.oshape == (4,)

    def test_shape_2d(self):
        assert Diff(axis=1, ishape=(3, 5)).oshape == (3, 4)

    def test_forward_values(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        np.testing.assert_array_equal(
            Diff(axis=0, ishape=(4,)).forward(x), np.array([2.0, 3.0, 4.0])
        )

    def test_dottest(self):
        assert dottest(Diff(axis=0, ishape=(8,)), num=5)

    def test_dottest_2d(self):
        assert dottest(Diff(axis=1, ishape=(4, 6)), num=5)


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


class TestSlice:
    def test_shape(self):
        op = Slice((10,), np.index_exp[::2])
        assert op.ishape == (10,)
        assert op.oshape == (5,)

    def test_forward(self):
        op = Slice((6,), np.index_exp[::2])
        np.testing.assert_array_equal(
            op.forward(np.arange(6.0)), np.array([0.0, 2.0, 4.0])
        )

    def test_dottest(self):
        assert dottest(Slice((10,), np.index_exp[::2]), num=5)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_forward_shape(self):
        idx = (np.array([0, 1, 2]),)
        op = Sampling(ishape=(5,), index=idx)
        assert op.forward(np.arange(5.0)).shape == (3,)

    def test_dottest(self):
        idx = (np.array([0, 2, 3, 1]),)
        assert dottest(Sampling(ishape=(6,), index=idx), num=5)


# ---------------------------------------------------------------------------
# CircConv
# ---------------------------------------------------------------------------


class TestCircConv:
    def test_shape(self):
        op = CircConv(np.array([1.0, 0.5, 0.0]), (8,))
        assert op.ishape == op.oshape == (8,)

    def test_dottest(self):
        assert dottest(CircConv(RNG.standard_normal(3), (8,)), num=5)

    def test_fwadjtest(self):
        assert fwadjtest(CircConv(RNG.standard_normal(3), (8,)), num=5)


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------


class TestConv:
    def test_shape_1d(self):
        op = Conv(np.ones(3), ishape=(8,), dim=1)
        assert op.ishape == (8,)
        assert op.oshape == (6,)  # 8 - 3 + 1

    def test_shape_2d(self):
        ir = np.ones((3, 3))
        op = Conv(ir, ishape=(8, 8), dim=2)
        assert op.oshape == (6, 6)

    def test_dottest_1d(self):
        assert dottest(Conv(RNG.standard_normal(3), ishape=(8,), dim=1), num=5)

    def test_dottest_2d(self):
        ir = RNG.standard_normal((3, 3))
        assert dottest(Conv(ir, ishape=(8, 8), dim=2), num=3)

    @pytest.mark.parametrize("kshape", [(3,), (4,), (5,), (2,)])
    def test_forward_is_valid_conv_1d(self, kshape):
        ir, x = RNG.standard_normal(kshape), RNG.standard_normal(12)
        out = Conv(ir, ishape=x.shape, dim=1).forward(x)
        np.testing.assert_allclose(out, np.convolve(x, ir, mode="valid"))

    @pytest.mark.parametrize("kshape", [(3, 3), (4, 4), (3, 4), (2, 5)])
    def test_forward_is_valid_conv_2d(self, kshape):
        ir, x = RNG.standard_normal(kshape), RNG.standard_normal((12, 13))
        out = Conv(ir, ishape=x.shape, dim=2).forward(x)
        np.testing.assert_allclose(out, valid_conv(x, ir))

    def test_forward_batch(self):
        ir, x = RNG.standard_normal((4, 4)), RNG.standard_normal((2, 12, 13))
        out = Conv(ir, ishape=x.shape, dim=2).forward(x)
        for b in range(2):
            np.testing.assert_allclose(out[b], valid_conv(x[b], ir))

    def test_kernel_per_channel(self):
        ir, x = RNG.standard_normal((2, 4, 3)), RNG.standard_normal((2, 12, 13))
        op = Conv(ir, ishape=x.shape, dim=2)
        assert op.oshape == (2, 9, 11)
        out = op.forward(x)
        for b in range(2):
            np.testing.assert_allclose(out[b], valid_conv(x[b], ir[b]))
        assert dottest(op, num=3)

    def test_kernel_broadcast_batch(self):
        ir, x = RNG.standard_normal((1, 3, 3)), RNG.standard_normal((2, 12, 12))
        op = Conv(ir, ishape=x.shape, dim=2)
        assert op.oshape == (2, 10, 10)
        np.testing.assert_allclose(
            op.forward(x), Conv(ir[0], ishape=x.shape, dim=2).forward(x)
        )
        assert dottest(op, num=3)

    @pytest.mark.parametrize(
        ("irshape", "ishape"),
        [
            ((3, 3, 3), (2, 12, 12)),  # batch 3 does not match 2
            ((2, 1, 3, 3), (2, 12, 12)),  # more batch axes than the input
            ((4, 3, 3), (12, 12)),  # batch axes but input has none
            ((3,), (12, 12)),  # fewer axes than `dim`
        ],
    )
    def test_invalid_kernel_batch(self, irshape, ishape):
        with pytest.raises(ValueError):
            Conv(RNG.standard_normal(irshape), ishape=ishape, dim=2)


def valid_conv(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Reference "valid" convolution: Σₖ h[k]·x[m + K-1 - k]."""
    windows = np.lib.stride_tricks.sliding_window_view(x, ir.shape)
    return np.tensordot(windows, np.flip(ir), axes=ir.ndim)
