import importlib.util

import numpy as np
import pytest
import udft
from conftest import RNG

from aljabr.concrete import (
    DFT,
    DWT,
    Analysis2,
    CircConv,
    Conv,
    Diag,
    Diff,
    DirectConv,
    FreqFilter,
    Identity,
    RealDFT,
    Sampling,
    Slice,
    Synthesis2,
)
from aljabr.utils import dottest, fwadjtest

needs_pywt = pytest.mark.skipif(
    importlib.util.find_spec("pywt") is None, reason="pywavelets not installed"
)

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

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    @pytest.mark.parametrize("axis", [0, -1])
    def test_adjoint_keeps_dtype(self, dtype, axis):
        op = Diff(axis=axis, ishape=(4, 6))
        y = RNG.standard_normal(op.oshape).astype(dtype)
        out = op.adjoint(y)
        assert out.dtype == dtype
        np.testing.assert_allclose(
            out,
            -np.diff(y.astype(np.complex128), prepend=0, append=0, axis=axis),
            rtol=1e-6,
        )


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

    def test_large_ishape_without_allocation(self):
        # 8 TB as float64: must not be allocated to infer oshape
        op = Slice((10**6, 10**6), np.index_exp[::2, 3])
        assert op.oshape == (500_000,)

    @pytest.mark.parametrize(
        "idx",
        [
            np.index_exp[1:4, ::-1],
            np.index_exp[..., 0],
            np.index_exp[None, 2],
            np.index_exp[[0, 2], 1:],
            (np.arange(4) % 2 == 0,),
        ],
    )
    def test_oshape_like_numpy(self, idx):
        x = np.zeros((4, 5))
        assert Slice(x.shape, idx).oshape == x[idx].shape


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

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    def test_adjoint_keeps_dtype(self, dtype):
        # Repeated index 0 to check accumulation
        op = Sampling(ishape=(4,), index=(np.array([0, 2, 0]),))
        y = np.array([1.0, 3.0, 5.0], dtype=dtype)
        if np.iscomplexobj(y):
            y = y + 1j * np.array([2.0, -1.0, 1.0], dtype=dtype)
        x = op.adjoint(y)
        assert x.dtype == dtype
        expected = np.zeros(4, dtype=dtype)
        np.add.at(expected, op.index, y)
        np.testing.assert_allclose(x, expected)

    def test_list_index(self):
        op = Sampling(ishape=(6,), index=([0, 2, 3],))
        assert op.oshape == (3,)
        np.testing.assert_array_equal(op.forward(np.arange(6.0)), [0.0, 2.0, 3.0])
        assert dottest(op, num=3)

    def test_broadcast_index(self):
        index = (np.array([[0], [2]]), np.array([1, 3, 4]))
        op = Sampling(ishape=(4, 6), index=index)
        assert op.oshape == (2, 3)
        x = RNG.standard_normal((4, 6))
        np.testing.assert_array_equal(op.forward(x), x[index])
        assert dottest(op, num=3)

    def test_negative_index(self):
        op = Sampling(ishape=(5,), index=(np.array([-1, 0, -1]),))
        x = np.arange(5.0)
        np.testing.assert_array_equal(op.forward(x), [4.0, 0.0, 4.0])
        np.testing.assert_array_equal(op.adjoint(np.ones(3)), [1.0, 0.0, 0.0, 0.0, 2.0])
        assert dottest(op, num=3)

    def test_index_count_raises(self):
        with pytest.raises(ValueError, match="one array per dimension"):
            Sampling(ishape=(4, 6), index=(np.array([0, 1]),))


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

    @pytest.mark.parametrize(
        ("ir_shape", "shape"), [((3,), (8, 10)), ((2, 3, 3), (8, 10))]
    )
    def test_ir_ndim_mismatch_raises(self, ir_shape, shape):
        with pytest.raises(ValueError, match="ndim"):
            CircConv(np.ones(ir_shape), shape)

    def test_1d_conv_of_2d_array(self):
        ir = RNG.standard_normal(3)
        x = RNG.standard_normal((4, 8))
        y = CircConv(ir[None, :], x.shape).forward(x)
        for row_x, row_y in zip(x, y):
            np.testing.assert_allclose(row_y, CircConv(ir, (8,)).forward(row_x))


# ---------------------------------------------------------------------------
# FreqFilter
# ---------------------------------------------------------------------------


class TestFreqFilter:
    def test_shape_is_half_spectrum(self):
        op = FreqFilter(np.ones((3, 3)) / 9, shape=(8, 10))
        assert op.ishape == op.oshape == (8, 6)

    def test_is_circconv_in_fourier_space(self):
        ir = RNG.standard_normal((3, 3))
        x = RNG.standard_normal((8, 10))
        op = FreqFilter(ir, x.shape)
        y = udft.irdftn(op.forward(udft.rdftn(x, 2)), x.shape)
        np.testing.assert_allclose(y, CircConv(ir, x.shape).forward(x))


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


# ---------------------------------------------------------------------------
# DirectConv
# ---------------------------------------------------------------------------


class TestDirectConv:
    def test_shape(self):
        op = DirectConv(np.ones((3, 4)), ishape=(8, 8))
        assert op.oshape == (6, 5)

    @pytest.mark.parametrize("kshape", [(3, 3), (4, 4), (2, 5)])
    def test_forward_is_valid_conv(self, kshape):
        ir, x = RNG.standard_normal(kshape), RNG.standard_normal((12, 13))
        out = DirectConv(ir, ishape=x.shape).forward(x)
        np.testing.assert_allclose(out, valid_conv(x, ir))

    def test_same_as_conv(self):
        ir, x = RNG.standard_normal((4, 3)), RNG.standard_normal((12, 13))
        np.testing.assert_allclose(
            DirectConv(ir, ishape=x.shape).forward(x),
            Conv(ir, ishape=x.shape, dim=2).forward(x),
        )

    def test_dottest(self):
        assert dottest(DirectConv(RNG.standard_normal((3, 3)), ishape=(8, 8)), num=3)

    def test_dottest_batch(self):
        assert dottest(DirectConv(RNG.standard_normal(3), ishape=(4, 8)), num=3)

    def test_ir_keeps_shape(self):
        ir = RNG.standard_normal((1, 3))
        assert DirectConv(ir, ishape=(4, 8, 8)).ir.shape == (1, 3)

    def test_dottest_complex_ir(self):
        ir = RNG.standard_normal((3, 3)) + 1j * RNG.standard_normal((3, 3))
        assert dottest(DirectConv(ir, ishape=(8, 8)), num=3)


def valid_conv(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Reference "valid" convolution: Σₖ h[k]·x[m + K-1 - k]."""
    windows = np.lib.stride_tricks.sliding_window_view(x, ir.shape)
    return np.tensordot(windows, np.flip(ir), axes=ir.ndim)


# ---------------------------------------------------------------------------
# Wavelets
# ---------------------------------------------------------------------------


@needs_pywt
class TestWavelets:
    @pytest.mark.parametrize("wavelet", ["haar", "db2", "sym4"])
    def test_dwt_orthogonal(self, wavelet):
        op = DWT((64,), level=2, wavelet=wavelet)
        assert dottest(op, num=3)
        assert fwadjtest(op, num=3)

    @pytest.mark.parametrize("wavelet", ["haar", "db2"])
    def test_swt_orthogonal(self, wavelet):
        assert dottest(Analysis2((32, 32), 2, wavelet), num=3)
        assert dottest(Synthesis2((32, 32), 2, wavelet), num=3)

    @pytest.mark.parametrize("shape", [(8,), (1, 8, 8), (2, 8, 8)])
    @pytest.mark.parametrize("make", [Analysis2, Synthesis2])
    def test_swt_not_2d_raises(self, make, shape):
        with pytest.raises(ValueError, match="must be 2D"):
            make(shape, 1)

    def test_cube2coeffs_inverts_coeffs2cube(self):
        op = Analysis2((8, 8), 2, "db2")
        cube = op.forward(RNG.standard_normal(op.ishape))
        np.testing.assert_array_equal(op.coeffs2cube(op.cube2coeffs(cube)), cube)

    @pytest.mark.parametrize("wavelet", ["haar", "db2"])
    def test_synthesis_irs(self, wavelet):
        op = Synthesis2((8, 8), 1, wavelet)
        irs = op.get_irs()
        for chan in range(op.ishape[0]):
            delta = np.zeros(op.ishape)
            delta[chan, 0, 0] = 1
            np.testing.assert_allclose(irs[chan], op.forward(delta), atol=1e-12)
        np.testing.assert_allclose(
            op.get_frs(), np.conj(op.analysis.get_frs()), atol=1e-12
        )

    @pytest.mark.parametrize("wavelet", ["bior2.2", "rbio3.3"])
    @pytest.mark.parametrize(
        "make",
        [
            lambda w: DWT((64,), level=2, wavelet=w),
            lambda w: Analysis2((32, 32), 2, w),
            lambda w: Synthesis2((32, 32), 2, w),
        ],
        ids=["DWT", "Analysis2", "Synthesis2"],
    )
    def test_non_orthogonal_raises(self, make, wavelet):
        with pytest.raises(ValueError, match="not orthogonal"):
            make(wavelet)

    @pytest.mark.parametrize(
        ("shape", "level"), [((12,), 2), ((10,), 1), ((48, 64), 4)]
    )
    def test_dwt_divisible_level(self, shape, level):
        op = DWT(shape, level=level)
        assert op.forward(RNG.standard_normal(shape)).shape == op.oshape
        assert dottest(op, num=3)
        assert fwadjtest(op, num=3)

    @pytest.mark.parametrize(("shape", "level"), [((10,), 2), ((9,), 1), ((12, 10), 2)])
    def test_dwt_non_divisible_level_raises(self, shape, level):
        with pytest.raises(ValueError, match="divisible"):
            DWT(shape, level=level)

    @pytest.mark.parametrize(
        ("shape", "level"), [((64,), 6), ((48,), 4), ((48, 64), 4), ((12,), 2)]
    )
    def test_dwt_default_level(self, shape, level):
        op = DWT(shape)
        assert op.lvl == level
        assert dottest(op, num=3)

    def test_dwt_default_level_odd_size_raises(self):
        with pytest.raises(ValueError, match="no decomposition level"):
            DWT((9,))


# ---------------------------------------------------------------------------
# Shapes given as lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda s: Identity(s),
        lambda s: DFT(s, 2),
        lambda s: RealDFT(s, 2),
        lambda s: Conv(np.ones((3, 3)), s, 2),
        lambda s: DirectConv(np.ones((3, 3)), s),
        lambda s: FreqFilter(np.ones((3, 3)), s),
        lambda s: CircConv(np.ones((3, 3)), s),
        lambda s: Diff(0, s),
        lambda s: Sampling(s, (np.array([0, 1]), np.array([2, 3]))),
        lambda s: Slice(s, np.index_exp[::2]),
        pytest.param(lambda s: DWT(s, 1), marks=needs_pywt),
        pytest.param(lambda s: Analysis2(s, 1), marks=needs_pywt),
        pytest.param(lambda s: Synthesis2(s, 1), marks=needs_pywt),
    ],
    ids=[
        "Identity",
        "DFT",
        "RealDFT",
        "Conv",
        "DirectConv",
        "FreqFilter",
        "CircConv",
        "Diff",
        "Sampling",
        "Slice",
        "DWT",
        "Analysis2",
        "Synthesis2",
    ],
)
def test_list_shape(make):
    from_list, from_tuple = make([8, 8]), make((8, 8))
    assert from_list.ishape == from_tuple.ishape
    assert from_list.oshape == from_tuple.oshape


# ---------------------------------------------------------------------------
# Kernel larger than the input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda ir, s: Conv(ir, s, 2),
        lambda ir, s: DirectConv(ir, s),
        lambda ir, s: CircConv(ir, s),
    ],
    ids=["Conv", "DirectConv", "CircConv"],
)
def test_kernel_larger_than_input_raises(make):
    with pytest.raises(ValueError, match="must not exceed"):
        make(np.ones((3, 5)), (8, 4))


@pytest.mark.parametrize(
    "make",
    [lambda ir, s: Conv(ir, s, 2), lambda ir, s: DirectConv(ir, s)],
    ids=["Conv", "DirectConv"],
)
def test_kernel_as_large_as_input(make):
    op = make(RNG.standard_normal((3, 4)), (8, 4))
    assert op.oshape == (6, 1)
    assert op.forward(RNG.standard_normal(op.ishape)).shape == op.oshape
    assert dottest(op, num=3)


# ---------------------------------------------------------------------------
# Complex impulse response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda ir: Conv(ir, (8, 8), 2),
        lambda ir: CircConv(ir, (8, 8)),
        lambda ir: FreqFilter(ir, (8, 8)),
    ],
    ids=["Conv", "CircConv", "FreqFilter"],
)
def test_complex_ir_raises(make):
    with pytest.raises(ValueError):
        make(RNG.standard_normal((3, 3)) + 1j * RNG.standard_normal((3, 3)))


def test_complex_circ_conv_on_full_spectrum():
    # The alternative to `CircConv` for a complex `ir`, given in its docstring
    shape = (8, 8)
    ir = RNG.standard_normal((3, 3)) + 1j * RNG.standard_normal((3, 3))
    F = DFT(shape, len(shape))
    op = F.H @ Diag(udft.ir2fr(ir, shape, onesided=False)) @ F
    assert dottest(op, num=3)
