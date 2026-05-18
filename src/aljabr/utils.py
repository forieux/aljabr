"""The ``utils`` module
====================

Utility functions for testing and analysing linear operators: dot test, adjoint
consistency test, symmetry and definiteness checks, condition number
estimation,...

"""

from numpy.random import standard_normal as randn
import numpy as np
import array_api_compat as arr_api

from .linop import LinOp, Array, asmatrix

__all__ = [
    "allclose",
    "dottest",
    "fwadjtest",
    "is_sym",
    "is_pos_def",
    "is_semi_pos_def",
    "is_neg_def",
    "is_semi_neg_def",
    "cond",
    "fcond",
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
    if xp == np:
        return np.allclose(a, b, rtol=rtol, atol=atol)
    return bool(xp.all(xp.abs(a - b) <= atol + rtol * xp.abs(b)))


def dottest(
    linop: LinOp,
    num: int = 1,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    echo: bool = False,
    xp=np,
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

    Returns
    -------
    bool
        True if all `num` tests pass.

    Notes
    -----
    Numpy is used for random number generation; arrays are converted to `xp`
    if `xp` is not numpy.
    """
    test = True
    for _ in range(num):
        vvec = xp.asarray(randn(linop.isize))
        uvec = xp.asarray(randn(linop.osize))
        # Use sum instead of more efficient vdot for compatibility with non-NumPy namespaces (e.g. torch)
        lhs = xp.sum(
            xp.conj(xp.reshape(linop.rmatvec(uvec), (-1,))) * xp.reshape(vvec, (-1,))
        )
        rhs = xp.sum(
            xp.conj(xp.reshape(uvec, (-1,))) * xp.reshape(linop.matvec(vvec), (-1,))
        )
        test = test and allclose(lhs, rhs, rtol=rtol, atol=atol)
        if echo:
            print(f"(Aᴴ·u)ᴴ·v = {lhs} ≈ {rhs} = uᴴ·(A·v)")
    return test


def fwadjtest(
    linop: LinOp,
    num: int = 1,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    echo: bool = False,
    xp=np,
) -> bool:
    """Test `fwadj` validity

    Verify the validity `fwadj` wrt. `forward` and `adjoint` methods with equality

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

    Returns
    -------
    bool
        True if all `num` tests pass.
    """
    test = True
    for _ in range(num):
        vvec = xp.asarray(randn(linop.ishape))
        i = linop.fwadj(vvec)
        j = linop.adjoint(linop.forward(vvec))
        close = allclose(i, j, rtol=rtol, atol=atol)
        test = test and close
        if echo:
            print(f"(Aᴴ·A)·v = {i} ≈ {j} = Aᴴ·(A·v)")
    return test


def is_sym(linop: Array | LinOp) -> bool:
    """Return True if `linop` is symmetric.

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
    xp = arr_api.get_namespace(mat)
    return mat.shape[0] == mat.shape[1] and allclose(xp.matrix_transpose(mat), mat)


def is_pos_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is positive definite.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.

    Returns
    -------
    bool

    Notes
    -----
    A positive definite matrix $M$ has strictly positive eigenvalues, but the
    converse is not true for non-symmetric matrices. The function therefore tests
    all eigenvalues of $M + M^T$: since $x^T M x = x^T \frac{M + M^T}{2} x$
    for any vector $x$, positive definiteness of $M$ is equivalent to positive
    definiteness of $M + M^T$.
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return bool(xp.all(xp.linalg.eigvals(mat + xp.matrix_transpose(mat)) > 0))


def is_semi_pos_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is semi positive definite.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.

    Returns
    -------
    bool

    See Also
    --------
    is_pos_def
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return bool(xp.all(xp.linalg.eigvals(mat + xp.matrix_transpose(mat)) >= 0))


def is_neg_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is negative definite.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.

    Returns
    -------
    bool

    See Also
    --------
    is_pos_def
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return bool(xp.all(xp.linalg.eigvals(mat + xp.matrix_transpose(mat)) < 0))


def is_semi_neg_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is semi negative definite.

    Parameters
    ----------
    linop : Array or LinOp
        The operator or matrix to test.

    Returns
    -------
    bool

    See Also
    --------
    is_pos_def
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return bool(xp.all(xp.linalg.eigvals(mat + xp.matrix_transpose(mat)) <= 0))


def cond(linop: Array | LinOp) -> float:
    """Return the condition number κ

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
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    eig = xp.abs(xp.linalg.eigvals(mat))
    return float(xp.max(eig) / xp.min(eig))


def fcond(linop: LinOp, tol: float = 0.1) -> float:
    """Estimate the condition number κ

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
    """
    try:
        import scipy.sparse.linalg
    except ImportError as e:
        raise ImportError("scipy is required for fcond") from e
    eig = scipy.sparse.linalg.eigsh(
        scipy.sparse.linalg.aslinearoperator(linop),
        k=2,
        return_eigenvectors=False,
        which="BE",
        tol=tol,
    )
    return np.max(np.abs(eig)) / np.min(np.abs(eig))
