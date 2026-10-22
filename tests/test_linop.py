from typing import Any, cast

import numpy as np
import pytest
from conftest import RNG, make_dense

from aljabr.linop import (
    Adjoint,
    BaseOp,
    Dense,
    HStack,
    LinOp,
    ProdOp,
    Scaled,
    Symmetric,
    VStack,
    asmatrix,
    is_scalar,
    unvectorize,
    vectorize,
)
from aljabr.utils import dottest, fwadjtest

# ---------------------------------------------------------------------------
# vectorize / unvectorize
# ---------------------------------------------------------------------------


class TestVectorize:
    def test_single_array(self):
        x = np.arange(6.0).reshape(2, 3)
        assert vectorize(x).shape == (6, 1)

    def test_list_of_arrays(self):
        v = vectorize([np.ones(3), np.ones(4)])
        assert v.shape == (7, 1)

    def test_unvectorize_roundtrip(self):
        shapes = [(3,), (4,), (2, 2)]
        arrays = [RNG.standard_normal(s) for s in shapes]
        restored = unvectorize(vectorize(arrays), shapes)
        for orig, rest in zip(arrays, restored):
            np.testing.assert_array_equal(orig, rest)


# ---------------------------------------------------------------------------
# LinOp shape properties
# ---------------------------------------------------------------------------


class TestLinOpShapes:
    def test_ishape_oshape(self):
        op = make_dense(4, 3)
        assert op.ishape == (3,)
        assert op.oshape == (4,)

    def test_isize_osize(self):
        op = Dense(np.ones((6, 4)), ishape=(2, 2), oshape=(2, 3))
        assert op.isize == 4
        assert op.osize == 6

    def test_shape_ndim(self):
        op = make_dense(4, 3)
        assert op.shape == (4, 3)
        assert op.ndim == 2

    def test_forward_output_shape(self):
        op = make_dense(4, 3)
        assert op.forward(np.ones(3)).shape == (4,)

    def test_adjoint_output_shape(self):
        op = make_dense(4, 3)
        assert op.adjoint(np.ones(4)).shape == (3,)

    def test_metadata_init_recorded(self):
        op = make_dense(4, 3)
        assert op.metadata["init"] is not None

    def test_metadata_forward_recorded(self):
        op = make_dense(4, 3)
        op.forward(np.ones(3))
        assert len(op.metadata["forward"]) == 1

    def test_metadata_bounded(self, monkeypatch):
        monkeypatch.setattr(LinOp, "max_timings", 3)
        op = make_dense(4, 3)
        for _ in range(5):
            op.forward(np.ones(3))
        assert len(op.metadata["forward"]) == 3

    def test_input_is_positional_only(self):
        # Subclasses are free to name the input, so it cannot be a keyword
        op = make_dense(4, 3)
        with pytest.raises(TypeError):
            op.forward(point=np.ones(3))

    def test_extra_argument(self):
        class Gain(LinOp):
            def forward(self, x, gain=1.0):
                return gain * x

            def adjoint(self, y, gain=1.0):
                return gain * y

        op = Gain((3,), (3,))
        x = np.ones(3)
        np.testing.assert_array_equal(op.forward(x, gain=2.0), 2 * x)
        np.testing.assert_array_equal(op.adjoint(x, gain=3.0), 3 * x)


# ---------------------------------------------------------------------------
# LinOp.asmatrix (generic implementation)
# ---------------------------------------------------------------------------


def make_fft(n: int) -> BaseOp:
    """Return the unitary 1D FFT as a BaseOp (complex output from real input)."""
    return BaseOp(
        forward=lambda x: np.fft.fft(x, norm="ortho"),
        adjoint=lambda y: np.fft.ifft(y, norm="ortho"),
        ishape=(n,),
        oshape=(n,),
    )


class TestLinOpAsmatrix:
    def test_real_operator_is_float64(self):
        op = BaseOp(lambda x: 2 * x, lambda y: 2 * y, ishape=(3,), oshape=(3,))
        mat = op.asmatrix()
        assert mat.dtype == np.float64
        np.testing.assert_array_equal(mat, 2 * np.eye(3))

    def test_non_square(self):
        op = BaseOp(lambda x: x[:2], lambda y: np.r_[y, 0], ishape=(3,), oshape=(2,))
        np.testing.assert_array_equal(op.asmatrix(), np.eye(2, 3))

    def test_complex_operator_keeps_imaginary_part(self):
        mat = make_fft(4).asmatrix()
        assert np.iscomplexobj(mat)
        np.testing.assert_allclose(mat, np.fft.fft(np.eye(4), axis=0, norm="ortho"))

    def test_complex_operator_adjoint(self):
        op = make_fft(4)
        np.testing.assert_allclose(op.H.asmatrix(), op.asmatrix().conj().T)

    def test_like_dtype(self):
        op = BaseOp(lambda x: 2 * x, lambda y: 2 * y, ishape=(3,), oshape=(3,))
        assert op.asmatrix(like=np.zeros(1, dtype=np.float32)).dtype == np.float32


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


class TestDense:
    def test_dottest(self):
        assert dottest(make_dense(4, 3), num=5)

    def test_asmatrix(self):
        mat = RNG.standard_normal((3, 4))
        np.testing.assert_allclose(asmatrix(Dense(mat, ishape=(4,), oshape=(3,))), mat)

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            Dense(np.ones((2, 3, 4)))

    def test_ishape_size_mismatch(self):
        with pytest.raises(ValueError):
            Dense(np.ones((2, 3)), ishape=(5,))

    def test_oshape_size_mismatch(self):
        with pytest.raises(ValueError):
            Dense(np.ones((2, 3)), oshape=(5,))


# ---------------------------------------------------------------------------
# BaseOp
# ---------------------------------------------------------------------------


class TestBaseOp:
    def test_from_callables_dottest(self):
        mat = RNG.standard_normal((3, 4))
        op = BaseOp(
            forward=lambda x: mat @ x,
            adjoint=lambda y: mat.T @ y,
            ishape=(4,),
            oshape=(3,),
        )
        assert dottest(op, num=5)

    def test_custom_fwadj(self):
        mat = RNG.standard_normal((3, 4))
        ata = mat.T @ mat
        op = BaseOp(
            forward=lambda x: mat @ x,
            adjoint=lambda y: mat.T @ y,
            fwadj=lambda x: ata @ x,
            ishape=(4,),
            oshape=(3,),
        )
        assert fwadjtest(op, num=5)


# ---------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------


class TestAdjoint:
    def test_involution(self):
        op = make_dense(4, 3)
        assert Adjoint(Adjoint(op)) is op

    def test_H_returns_adjoint(self):
        op = make_dense(4, 3)
        adj = op.H
        assert isinstance(adj, Adjoint)

    def test_H_H_returns_self(self):
        op = make_dense(4, 3)
        assert op.H.H is op

    def test_shapes_swapped(self):
        op = make_dense(4, 3)
        adj = op.H
        assert adj.ishape == op.oshape
        assert adj.oshape == op.ishape

    def test_asmatrix_is_conjugate_transpose(self):
        mat = RNG.standard_normal((3, 4))
        op = Dense(mat, ishape=(4,), oshape=(3,))
        np.testing.assert_allclose(asmatrix(op.H), mat.T.conj())

    def test_dottest(self):
        assert dottest(make_dense(4, 3).H, num=5)

    def test_symmetric_H_is_self(self):
        op = make_dense(4, 3)
        sym = op.G
        assert sym.H is sym


# ---------------------------------------------------------------------------
# Symmetric (Gram)
# ---------------------------------------------------------------------------


class TestSymmetric:
    def test_gram_shape(self):
        op = make_dense(4, 3)
        gram = op.G
        assert gram.ishape == op.ishape
        assert gram.oshape == op.ishape

    def test_gram_H_is_self(self):
        op = make_dense(4, 3)
        gram = op.G
        assert gram.H is gram

    def test_fwadjtest(self):
        assert fwadjtest(make_dense(4, 3), num=5)

    def test_gram_equals_AH_at_A(self):
        op = make_dense(4, 3)
        np.testing.assert_allclose(asmatrix(op.G), asmatrix(op.H @ op), atol=1e-12)

    def test_matmul_detects_AH_A(self):
        op = make_dense(4, 3)
        assert isinstance(op.H @ op, Symmetric)

    def test_matmul_detects_A_AH(self):
        op = make_dense(4, 3)
        assert isinstance(op @ op.H, Symmetric)


# ---------------------------------------------------------------------------
# Scaled
# ---------------------------------------------------------------------------


class TestScaled:
    def test_scalar_mul_returns_scaled(self):
        assert isinstance(2.0 * make_dense(4, 3), Scaled)

    def test_forward_scaled(self):
        op = make_dense(4, 3)
        x = RNG.standard_normal(3)
        np.testing.assert_allclose((2.0 * op).forward(x), 2.0 * op.forward(x))

    def test_adjoint_conjugate_scale(self):
        op = make_dense(4, 3)
        s = 2 + 3j
        y = RNG.standard_normal(4)
        np.testing.assert_allclose((s * op).adjoint(y), s.conjugate() * op.adjoint(y))

    def test_fwadj_abs_squared_scale(self):
        op = make_dense(4, 3)
        s = 3.0
        x = RNG.standard_normal(3)
        np.testing.assert_allclose((s * op).fwadj(x), s**2 * op.fwadj(x))

    def test_dottest(self):
        assert dottest(3.0 * make_dense(4, 3), num=5)


# ---------------------------------------------------------------------------
# ProdOp
# ---------------------------------------------------------------------------


class TestProdOp:
    def test_shape(self):
        prod = make_dense(4, 3) @ make_dense(3, 2)
        assert prod.ishape == (2,)
        assert prod.oshape == (4,)

    def test_incompatible_shapes(self):
        with pytest.raises(ValueError):
            ProdOp(make_dense(4, 3), make_dense(5, 2))

    def test_matmul_operator(self):
        assert isinstance(make_dense(4, 3) @ make_dense(3, 2), ProdOp)

    def test_asmatrix(self):
        mat_a = RNG.standard_normal((4, 3))
        mat_b = RNG.standard_normal((3, 2))
        a = Dense(mat_a, ishape=(3,), oshape=(4,))
        b = Dense(mat_b, ishape=(2,), oshape=(3,))
        np.testing.assert_allclose(asmatrix(a @ b), mat_a @ mat_b, atol=1e-12)

    def test_dottest(self):
        assert dottest(make_dense(4, 3) @ make_dense(3, 2), num=5)


# ---------------------------------------------------------------------------
# AddOp / SubOp
# ---------------------------------------------------------------------------


class TestAddSubOp:
    def test_add_shape(self):
        a, b = make_dense(4, 3), make_dense(4, 3)
        assert (a + b).ishape == (3,)
        assert (a + b).oshape == (4,)

    def test_add_incompatible(self):
        with pytest.raises(ValueError):
            make_dense(4, 3) + make_dense(4, 2)

    def test_add_asmatrix(self):
        mat_a, mat_b = RNG.standard_normal((3, 4)), RNG.standard_normal((3, 4))
        a = Dense(mat_a, ishape=(4,), oshape=(3,))
        b = Dense(mat_b, ishape=(4,), oshape=(3,))
        np.testing.assert_allclose(asmatrix(a + b), mat_a + mat_b, atol=1e-12)

    def test_sub_asmatrix(self):
        mat_a, mat_b = RNG.standard_normal((3, 4)), RNG.standard_normal((3, 4))
        a = Dense(mat_a, ishape=(4,), oshape=(3,))
        b = Dense(mat_b, ishape=(4,), oshape=(3,))
        np.testing.assert_allclose(asmatrix(a - b), mat_a - mat_b, atol=1e-12)

    def test_add_dottest(self):
        assert dottest(make_dense(4, 3) + make_dense(4, 3), num=5)

    def test_sub_dottest(self):
        assert dottest(make_dense(4, 3) - make_dense(4, 3), num=5)

    @pytest.mark.parametrize("operand", [3, 2.0, "x", np.ones((4, 3))])
    def test_non_linop_operand_raises(self, operand):
        a = make_dense(4, 3)
        with pytest.raises(TypeError):
            a + operand
        with pytest.raises(TypeError):
            a - operand

    def test_duck_operand_raises(self):
        # Only LinOp subclasses compose: an object merely exposing the LinOp
        # methods and attributes is rejected.
        class Duck:
            ishape = (3,)
            oshape = (4,)

            def forward(self, point):
                return point

            def adjoint(self, point):
                return point

            def fwadj(self, point):
                return point

        a, duck = make_dense(4, 3), cast(Any, Duck())
        with pytest.raises(TypeError):
            a + duck
        with pytest.raises(TypeError):
            a - duck


# ---------------------------------------------------------------------------
# VStack / HStack
# ---------------------------------------------------------------------------


class TestVStack:
    def test_shape(self):
        v = VStack([make_dense(2, 3), make_dense(4, 3)])
        assert v.ishape == (3,)
        assert v.oshape == (6, 1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            VStack([])

    def test_incompatible_ishape(self):
        with pytest.raises(ValueError):
            VStack([make_dense(2, 3), make_dense(2, 4)])

    def test_H_is_HStack(self):
        v = VStack([make_dense(2, 3), make_dense(4, 3)])
        assert isinstance(v.H, HStack)

    def test_H_H_oplist(self):
        a, b = make_dense(2, 3), make_dense(4, 3)
        v = VStack([a, b])
        vhh = v.H.H
        assert vhh.oplist[0] is a
        assert vhh.oplist[1] is b

    def test_H_H_is_self(self):
        v = VStack([make_dense(2, 3), make_dense(4, 3)])
        assert v.H.H is v
        assert v.H.H.H is v.H

    def test_HStack_H_H_is_self(self):
        h = HStack([make_dense(3, 2), make_dense(3, 4)])
        assert h.H.H is h
        assert h.H.H.H is h.H

    def test_split_shapes(self):
        a, b = make_dense(2, 3), make_dense(4, 3)
        v = VStack([a, b])
        y = v.forward(RNG.standard_normal(3))
        parts = v.split(y)
        assert parts[0].shape == a.oshape
        assert parts[1].shape == b.oshape

    def test_dottest(self):
        assert dottest(VStack([make_dense(2, 3), make_dense(4, 3)]), num=5)


class TestHStack:
    def test_shape(self):
        h = HStack([make_dense(3, 2), make_dense(3, 4)])
        assert h.oshape == (3,)
        assert h.ishape == (6, 1)

    def test_H_is_VStack(self):
        h = HStack([make_dense(3, 2), make_dense(3, 4)])
        assert isinstance(h.H, VStack)

    def test_dottest(self):
        assert dottest(HStack([make_dense(3, 2), make_dense(3, 4)]), num=5)


# Python, NumPy and 0-d array scalars that must scale an operator
SCALARS = (
    2,
    2.0,
    2 + 1j,
    np.float32(2),
    np.float64(2),
    np.int64(2),
    np.complex64(2 + 1j),
    np.array(2.0),
)


# ---------------------------------------------------------------------------
# Operator overloads
# ---------------------------------------------------------------------------


class TestOperatorOverloads:
    def test_mul_array_is_forward(self):
        op = make_dense(4, 3)
        x = RNG.standard_normal(3)
        np.testing.assert_array_equal(op * x, op.forward(x))

    def test_rmul_array_is_adjoint(self):
        op = make_dense(4, 3)
        y = RNG.standard_normal(4)
        np.testing.assert_array_equal(y * op, op.adjoint(y))

    def test_rmul_scalar_is_scaled(self):
        assert isinstance(2.0 * make_dense(4, 3), Scaled)

    @pytest.mark.parametrize("scalar", SCALARS, ids=repr)
    def test_scalar_left_and_right_is_scaled(self, scalar):
        op = make_dense(4, 3)
        x = RNG.standard_normal(3)
        for scaled in (scalar * op, op * scalar, scalar @ op, op @ scalar):
            assert isinstance(scaled, Scaled)
            np.testing.assert_allclose(scaled.forward(x), scalar * op.forward(x))
        assert dottest(scalar * op, num=3)

    @pytest.mark.parametrize("scalar", SCALARS, ids=repr)
    def test_is_scalar(self, scalar):
        assert is_scalar(scalar)
        assert is_scalar(scalar, shape=(3,))

    @pytest.mark.parametrize("value", [np.ones(1), np.ones((2, 2)), [2.0], "2"])
    def test_is_not_scalar(self, value):
        assert not is_scalar(value)

    def test_zero_dim_array_is_not_scalar_for_zero_dim_shape(self):
        assert not is_scalar(np.array(2.0), shape=())
        assert is_scalar(2.0, shape=())

    def test_zero_dim_array_is_operand_for_zero_dim_operator(self):
        # An operator on 0-d arrays applies to them instead of being scaled
        op = BaseOp(lambda x: 2 * x, lambda y: 3 * y, ishape=(), oshape=())
        assert op * np.array(5.0) == 10.0
        assert np.array(5.0) * op == 15.0

    def test_matmul_column_is_matvec(self):
        op = make_dense(4, 3)
        x = RNG.standard_normal((3, 1))
        np.testing.assert_array_equal(op @ x, op.matvec(x))

    def test_matmul_linop_is_prodop(self):
        assert isinstance(make_dense(4, 3) @ make_dense(3, 2), ProdOp)

    def test_call_is_forward(self):
        op = make_dense(4, 3)
        x = RNG.standard_normal(3)
        np.testing.assert_array_equal(op(x), op.forward(x))
