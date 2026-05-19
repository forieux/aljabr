import numpy as np
import pytest

from aljabr.concrete import (
    CircConv,
    Conv,
    Diag,
    DFT,
    Diff,
    Identity,
    RealDFT,
    Sampling,
    Slice,
)
from aljabr.utils import dottest, fwadjtest

from conftest import RNG


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
