# Copyright (c) 2026 François Orieux <francois.orieux@universite-paris-saclay.fr>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the " Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The ``utils`` module
====================

Utility functions for testing and analyzing linear operators: dot test, adjoint
consistency test, symmetry and definiteness checks, condition number
estimation,...

"""

import types
import warnings

import array_api_compat as arr_api
import numpy as np
from numpy.random import standard_normal as randn

from .linop import Array, LinOp, asmatrix

splinalg: types.ModuleType | None

try:
    import scipy.sparse.linalg as splinalg
except ImportError:
    splinalg = None

__all__ = [
    "allclose",
    "cond",
    "dottest",
    "fcond",
    "fwadjtest",
    "herm",
    "is_neg_def",
    "is_pos_def",
    "is_semi_neg_def",
    "is_semi_pos_def",
    "is_sym",
]


def allclose(a: Array, b: Array, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Array-namespace agnostic equivalent of `np.allclose`.

    Works for scalars (0-d arrays) and multi-dimensional arrays alike.

    Parameters
    ----------
    a, b : Array
        Arrays to compare.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    bool
        True if all elements satisfy ``|a - b| <= atol + rtol * |b|``.
    """
    xp = arr_api.get_namespace(a)
    if xp is np:
        return np.allclose(a, b, rtol=rtol, atol=atol)
    return bool(xp.all(xp.abs(a - b) <= atol + rtol * xp.abs(b)))


def dottest(
    linop: LinOp,
    num: int = 1,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    echo: bool = False,
    xp=np,
    device=None,
    complex_vectors: bool = False,
) -> bool:
    """The dot test.

    Verify the validity of `forward` and `adjoint` methods with equality

    `(Aᴴ·u)ᴴ·v = uᴴ·(A·v)`.

    where `u` and `v` are random vectors, to detect errors in implementation.

    Parameters
    ----------
    linop : LinOp
        The linear operator to test.
    num : int, optional
        The number of tests. They must all pass.
    rtol : float, optional
        The relative tolerance parameter (see `allclose`).
    atol : float, optional
        The absolute tolerance parameter (see `allclose`).
    echo : bool, optional
        If True, print the two sides of the equality for each test.
    xp : array namespace, optional
        The array namespace for generating test vectors (default: numpy).
    device : device, optional
        The device of the test vectors, as in `xp.asarray` (default: the
        default device of `xp`).
    complex_vectors : bool, optional
        If True, the test vectors are complex, to also test the operator on
        the imaginary part of its input. Default to real vectors, since some
        operators accept only real input.

    Returns
    -------
    bool
        True if all `num` tests pass.

    Warns
    -----
    UserWarning
        For each failed test, with the two sides of the equality.

    Notes
    -----
    Numpy is used for random number generation; arrays are converted to `xp`
    if `xp` is not numpy.
    """
    test = True
    for _ in range(num):
        vvec = randn(linop.isize)
        uvec = randn(linop.osize)
        if complex_vectors:
            # Not `+=`: in place, NumPy can not cast a complex sum to float
            vvec = vvec + 1j * randn(linop.isize)
            uvec = uvec + 1j * randn(linop.osize)
        vvec = xp.asarray(vvec, device=device)
        uvec = xp.asarray(uvec, device=device)
        # Use sum instead of more efficient vdot for compatibility with non-NumPy namespaces (e.g. torch)
        lhs = xp.sum(
            xp.conj(xp.reshape(linop.rmatvec(uvec), (-1,))) * xp.reshape(vvec, (-1,))
        )
        rhs = xp.sum(
            xp.conj(xp.reshape(uvec, (-1,))) * xp.reshape(linop.matvec(vvec), (-1,))
        )
        close = allclose(lhs, rhs, rtol=rtol, atol=atol)
        test = test and close
        if echo:
            print(f"(Aᴴ·u)ᴴ·v = {lhs} ≈ {rhs} = uᴴ·(A·v)")
        if not close:
            warnings.warn(
                f"dot test failed for {linop!r}. (Aᴴ·u)ᴴ·v = {complex(lhs)} "
                f"≠ {complex(rhs)} = uᴴ·(A·v), relative error "
                f"{relerr(lhs, rhs):.2e}",
                stacklevel=2,
            )
    return test


def fwadjtest(
    linop: LinOp,
    num: int = 1,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    echo: bool = False,
    xp=np,
    device=None,
    complex_vectors: bool = False,
) -> bool:
    """Test `fwadj` validity.

    Verify the validity of `fwadj` wrt. `forward` and `adjoint` methods with equality

    `(Aᴴ·A)·v = Aᴴ·(A·v)`.

    where `v` is a random vector, to detect errors in implementation.

    Parameters
    ----------
    linop : LinOp
        The linear operator to test.
    num : int, optional
        The number of tests. They must all pass.
    rtol : float, optional
        The relative tolerance parameter (see `allclose`).
    atol : float, optional
        The absolute tolerance parameter (see `allclose`).
    echo : bool, optional
        If True, print the two sides of the equality for each test.
    xp : array namespace, optional
        The array namespace for generating test vectors (default: numpy).
    device : device, optional
        The device of the test vectors, as in `xp.asarray` (default: the
        default device of `xp`).
    complex_vectors : bool, optional
        If True, the test vectors are complex, to also test the operator on
        the imaginary part of its input. Default to real vectors, since some
        operators accept only real input.

    Returns
    -------
    bool
        True if all `num` tests pass.

    Warns
    -----
    UserWarning
        For each failed test, with the relative error.
    """
    test = True
    for _ in range(num):
        vvec = randn(linop.ishape)
        if complex_vectors:
            vvec = vvec + 1j * randn(linop.ishape)
        vvec = xp.asarray(vvec, device=device)
        i = linop.fwadj(vvec)
        j = linop.adjoint(linop.forward(vvec))
        close = allclose(i, j, rtol=rtol, atol=atol)
        test = test and close
        if echo:
            print(f"(Aᴴ·A)·v = {i} ≈ {j} = Aᴴ·(A·v)")
        if not close:
            warnings.warn(
                f"fwadj test failed for {linop!r}. (Aᴴ·A)·v ≠ Aᴴ·(A·v), "
                f"relative error {relerr(i, j):.2e} (max norm)",
                stacklevel=2,
            )
    return test


def relerr(a: Array, b: Array) -> float:
    """Return `max|a - b| / max|b|` (`inf` if `b` is zero), for messages."""
    xp = arr_api.get_namespace(a, b)
    err = float(xp.max(xp.abs(a - b)))
    ref = float(xp.max(xp.abs(b)))
    return err / ref if ref else float("inf")


def herm(mat: Array) -> Array:
    r"""Return the conjugate transpose `Mᴴ` of a matrix.

    Its coefficients are

    .. math::

        (M^H)_{ij} = \overline{M_{ji}}.

    Parameters
    ----------
    mat : Array
        A matrix, or a stack of matrices on the last two axes.

    Returns
    -------
    Array
        The conjugate transpose, equal to the transpose for a real matrix.
    """
    xp = arr_api.get_namespace(mat)
    return xp.conj(xp.matrix_transpose(mat))


def is_sym(linop: Array | LinOp) -> bool:
    r"""Return True if `linop` is symmetric, that is `M = Mᴴ`.

    For a complex matrix, this is the Hermitian property (as for `Symmetric`)

    .. math::

        M_{ij} = \overline{M_{ji}} \quad \forall i, j,

    or equivalently :math:`y^H (M x) = (M y)^H x` for all vectors :math:`x` and
    :math:`y`.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.

    Returns
    -------
    bool

    See Also
    --------
    scipy.linalg.issymmetric
    scipy.linalg.ishermitian
    """
    mat = asmatrix(linop)
    return mat.shape[0] == mat.shape[1] and allclose(herm(mat), mat)


def _herm_part_eigvals(linop: Array | LinOp, rtol: float | None) -> tuple:
    """Return the eigenvalues of `M + Mᴴ` and the absolute tolerance on them."""
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    eig = xp.linalg.eigvalsh(mat + herm(mat))
    if rtol is None:
        rtol = max(mat.shape) * xp.finfo(eig.dtype).eps
    return eig, rtol * xp.max(xp.abs(eig))


def is_pos_def(linop: Array | LinOp, rtol: float | None = None) -> bool:
    r"""Return True if `linop` is positive definite.

    That is

    .. math::

        \Re(x^H M x) > 0 \quad \forall x \neq 0
        \iff \lambda_{\min}(M + M^H) > 0.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.
    rtol : float, optional
        Eigenvalues `λ` of `M + Mᴴ` must satisfy `λ > rtol·max(|λ|)`. Defaults
        to `max(M.shape)·eps`, as `numpy.linalg.matrix_rank`.

    Returns
    -------
    bool

    Notes
    -----
    The eigenvalues of a positive definite matrix :math:`M` have a strictly
    positive real part, but the converse is not true for non-Hermitian
    matrices. The function therefore tests all eigenvalues of :math:`M + M^H`:
    since

    .. math::

        \Re(x^H M x) = x^H \frac{M + M^H}{2} x

    for any vector :math:`x`, the condition :math:`\Re(x^H M x) > 0` is
    equivalent to the positive definiteness of the Hermitian matrix
    :math:`M + M^H`. For a Hermitian :math:`M`, this is the usual definition.
    """
    eig, tol = _herm_part_eigvals(linop, rtol)
    xp = arr_api.get_namespace(eig)
    return bool(xp.all(eig > tol))


def is_semi_pos_def(linop: Array | LinOp, rtol: float | None = None) -> bool:
    r"""Return True if `linop` is semi positive definite.

    That is

    .. math::

        \Re(x^H M x) \geq 0 \quad \forall x
        \iff \lambda_{\min}(M + M^H) \geq 0.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.
    rtol : float, optional
        Eigenvalues `λ` of `M + Mᴴ` must satisfy `λ ≥ -rtol·max(|λ|)`. See
        `is_pos_def`.

    Returns
    -------
    bool

    See Also
    --------
    is_pos_def
    """
    eig, tol = _herm_part_eigvals(linop, rtol)
    xp = arr_api.get_namespace(eig)
    return bool(xp.all(eig >= -tol))


def is_neg_def(linop: Array | LinOp, rtol: float | None = None) -> bool:
    r"""Return True if `linop` is negative definite.

    That is

    .. math::

        \Re(x^H M x) < 0 \quad \forall x \neq 0
        \iff \lambda_{\max}(M + M^H) < 0.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.
    rtol : float, optional
        Eigenvalues `λ` of `M + Mᴴ` must satisfy `λ < -rtol·max(|λ|)`. See
        `is_pos_def`.

    Returns
    -------
    bool

    See Also
    --------
    is_pos_def
    """
    eig, tol = _herm_part_eigvals(linop, rtol)
    xp = arr_api.get_namespace(eig)
    return bool(xp.all(eig < -tol))


def is_semi_neg_def(linop: Array | LinOp, rtol: float | None = None) -> bool:
    r"""Return True if `linop` is semi negative definite.

    That is

    .. math::

        \Re(x^H M x) \leq 0 \quad \forall x
        \iff \lambda_{\max}(M + M^H) \leq 0.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.
    rtol : float, optional
        Eigenvalues `λ` of `M + Mᴴ` must satisfy `λ ≤ rtol·max(|λ|)`. See
        `is_pos_def`.

    Returns
    -------
    bool

    See Also
    --------
    is_pos_def
    """
    eig, tol = _herm_part_eigvals(linop, rtol)
    xp = arr_api.get_namespace(eig)
    return bool(xp.all(eig <= tol))


def cond(linop: Array | LinOp) -> float:
    """Return the condition number κ.

    The condition number κ is defined as::

        κ = max(|λ|) / min(|λ|)

    where λ are eigenvalues of `linop`.

    Parameters
    ----------
    linop : Array or LinOp
        An implicit linear operator or a matrix.

    Returns
    -------
    float
        The condition number.

    Notes
    -----
    This is the usual condition number `σ_max / σ_min` (ratio of eigen values)
    only for normal matrices.

    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    eig = xp.abs(xp.linalg.eigvals(mat))
    return float(xp.max(eig) / xp.min(eig))


def fcond(linop: LinOp, tol: float = 0.1) -> float:
    """Estimate the condition number κ.

    The condition number κ is defined as::

        κ = max(|λ|) / min(|λ|)

    where the two extreme eigenvalues λ of `linop` are estimated with the
    Lanczos algorithm via `scipy.sparse.linalg.eigsh`.

    Parameters
    ----------
    linop : LinOp
        An implicit linear operator.
    tol : float, optional
        Tolerance for `scipy.sparse.linalg.eigsh`.

    Returns
    -------
    float
        Estimated condition number.

    Notes
    -----
    `eigsh` assumes that `linop` is Hermitian and `which="BE"` returns the
    algebraically smallest and largest eigenvalues. The estimate is therefore
    valid for Hermitian definite operators, not for indefinite ones.
    """
    if splinalg is None:
        raise ImportError("scipy is required for fcond")

    if linop.isize <= 2:
        eig = np.linalg.eigvalsh(np.asarray(asmatrix(linop)))
    else:
        eig = splinalg.eigsh(
            splinalg.aslinearoperator(linop),
            k=2,
            return_eigenvectors=False,
            which="BE",
            tol=tol,
        )
    return np.max(np.abs(eig)) / np.min(np.abs(eig))
