import numpy as np
import pytest
from conftest import RNG

from aljabr.concrete import DFT, Diag, Diff
from aljabr.utils import (
    cond,
    fcond,
    herm,
    is_neg_def,
    is_pos_def,
    is_semi_neg_def,
    is_semi_pos_def,
    is_sym,
)

# Hermitian with eigenvalues -1 and 3: indefinite, but M + Mᵀ = 2I
HERM_INDEF = np.array([[1, 2j], [-2j, 1]])
HERM_POS = np.array([[2, 1j], [-1j, 2]])


# ---------------------------------------------------------------------------
# herm
# ---------------------------------------------------------------------------


class TestHerm:
    def test_real_is_transpose(self):
        mat = RNG.standard_normal((3, 4))
        np.testing.assert_array_equal(herm(mat), mat.T)

    def test_complex_is_conjugate_transpose(self):
        mat = RNG.standard_normal((3, 4)) + 1j * RNG.standard_normal((3, 4))
        np.testing.assert_array_equal(herm(mat), mat.conj().T)


# ---------------------------------------------------------------------------
# is_sym
# ---------------------------------------------------------------------------


class TestIsSym:
    def test_real_symmetric(self):
        mat = RNG.standard_normal((4, 4))
        assert is_sym(mat + mat.T)

    def test_real_non_symmetric(self):
        assert not is_sym(np.array([[1.0, 2.0], [0.0, 1.0]]))

    def test_non_square(self):
        assert not is_sym(np.ones((2, 3)))

    def test_hermitian(self):
        assert is_sym(HERM_POS)

    def test_complex_symmetric_is_not_hermitian(self):
        assert not is_sym(np.array([[2, 1j], [1j, 2]]))

    def test_gram_operator(self):
        assert is_sym(Diff(0, (5,)).G)
        assert is_sym(DFT((4, 4), 2).G)


# ---------------------------------------------------------------------------
# is_pos_def, is_semi_pos_def, is_neg_def, is_semi_neg_def
# ---------------------------------------------------------------------------


class TestDefiniteness:
    def test_pos_def(self):
        assert is_pos_def(np.diag([1.0, 2.0]))
        assert not is_pos_def(np.diag([1.0, -2.0]))

    def test_neg_def(self):
        assert is_neg_def(np.diag([-1.0, -2.0]))
        assert not is_neg_def(np.diag([1.0, -2.0]))

    def test_real_non_symmetric(self):
        # xᵀMx = x₁² + x₂² > 0 although M is not symmetric
        assert is_pos_def(np.array([[1.0, 1.0], [-1.0, 1.0]]))

    def test_hermitian_pos_def(self):
        assert is_pos_def(HERM_POS)

    def test_hermitian_indefinite(self):
        assert not is_pos_def(HERM_INDEF)
        assert not is_semi_pos_def(HERM_INDEF)
        assert not is_neg_def(HERM_INDEF)
        assert not is_semi_neg_def(HERM_INDEF)

    @pytest.mark.parametrize("size", [5, 16, 33, 64])
    def test_semi_def_with_zero_eigenvalue(self, size):
        # The constant vector is in the kernel of DᴴD
        gram = Diff(0, (size,)).G
        assert is_semi_pos_def(gram)
        assert not is_pos_def(gram)
        assert is_semi_neg_def(-1.0 * gram)
        assert not is_neg_def(-1.0 * gram)

    def test_rtol(self):
        mat = np.diag([1.0, -1e-6])
        assert not is_semi_pos_def(mat)
        assert is_semi_pos_def(mat, rtol=1e-5)

    def test_zero(self):
        assert is_semi_pos_def(np.zeros((2, 2)))
        assert not is_pos_def(np.zeros((2, 2)))

    def test_operator(self):
        assert is_pos_def(Diag(np.array([1.0, 2.0, 3.0])))


# ---------------------------------------------------------------------------
# cond
# ---------------------------------------------------------------------------


class TestCond:
    def test_diag(self):
        assert cond(Diag(np.array([1.0, 4.0, -2.0]))) == pytest.approx(4.0)

    def test_unitary(self):
        assert cond(DFT((8,), 1)) == pytest.approx(1.0)


class TestFcond:
    def test_lanczos(self):
        diag = np.linspace(1.0, 10.0, 20)
        assert fcond(Diag(diag), tol=1e-8) == pytest.approx(10.0)

    @pytest.mark.parametrize(("diag", "expected"), [([1.0, 4.0], 4.0), ([3.0], 1.0)])
    def test_small(self, diag, expected):
        assert fcond(Diag(np.array(diag))) == pytest.approx(expected)
