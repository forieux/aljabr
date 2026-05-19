import numpy as np
import pytest

from aljabr.linop import (
    AddOp,
    Adjoint,
    BaseOp,
    Dense,
    HStack,
    ProdOp,
    Scaled,
    SubOp,
    Symmetric,
    VStack,
    asmatrix,
    unvectorize,
    vectorize,
)
from aljabr.utils import dottest, fwadjtest

from conftest import RNG, make_dense


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
        np.testing.assert_allclose(
            asmatrix(op.G), asmatrix(op.H @ op), atol=1e-12
        )

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
