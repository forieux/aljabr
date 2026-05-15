from numpy.random import standard_normal as randn
import numpy as np
import array_api_compat as arr_api

from .linop import LinOp, Array, asmatrix


def allclose(a, b, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Array-namespace agnostic equivalent of `np.allclose`.

    Works for scalars (0-d arrays) and multi-dimensional arrays alike.
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
        The number of test. They must all pass.
    rtol : float, optional
        The relative tolerance parameter (see np.allclose).
    atol : float, optional
        The absolute tolerance parameter (see np.allclose).
    xp : array namespace, optional
        The array namespace to use for generating test vectors (default: numpy).
        Pass the appropriate namespace for non-NumPy LinOps (e.g. torch).

    Comment
    -------
    Numpy is still use for random number generation and conversion are done if
    xp is not numpy.

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

    where `v` is a random vectors, to detect errors in implementation.

    Parameters
    ----------
    linop : LinOp
        The linear operator to test.
    num : int, optional
        The number of test. They must all pass.
    rtol : float, optional
        The relative tolerance parameter (see np.allclose).
    atol : float, optional
        The absolute tolerance parameter (see np.allclose).
    xp : array namespace, optional
        The array namespace to use for generating test vectors (default: numpy).
        Pass the appropriate namespace for non-NumPy LinOps (e.g. torch).
    """
    test = True
    for _ in range(num):
        vvec = xp.asarray(randn(linop.ishape))
        i = linop.fwadj(vvec)
        j = linop.adjoint(linop.forward(vvec))
        # Use all instead of allclose for compatibility with non-NumPy namespaces (e.g. torch)
        close = allclose(i, j, rtol=rtol, atol=atol)
        test = test and close
        if echo:
            print(f"(Aᴴ·A)·v = {i} ≈ {j} = Aᴴ·(A·v)")
    return test


def is_sym(linop: Array | LinOp) -> bool:
    """Return True if `linop` is symmetric

    See also
    --------
    - scipy.linalg.issymmetric
    - scipy.linalg.ishermitian
    """
    linop = asmatrix(linop)
    return linop.shape[0] == linop.shape[1] and allclose(linop.T, linop)


def is_pos_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is positive definite

    Notes
    -----

    Definite positive matrix $M$ implies that eigen values are strictly
    positives but inverse is not true. The function test that $M$ is symmetric
    and that all eigen values of $M^T + M$ are positives.`
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return is_sym(mat) and xp.all(xp.linalg.eigvals(mat + mat.T) > 0)


def is_semi_pos_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is semi positive definite

    Notes
    -----
    See :func:`is_pos_def`.
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return is_sym(mat) and xp.all(xp.linalg.eigvals(mat + mat.T) >= 0)


def is_neg_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is negative definite

    Notes
    -----
    See :func:`is_pos_def`.
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return is_sym(mat) and xp.all(xp.linalg.eigvals(mat + mat.T) < 0)


def is_semi_neg_def(linop: Array | LinOp) -> bool:
    """Return True if `linop` is semi negative definite

    Notes
    -----
    See :func:`is_pos_def`.
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    return is_sym(mat) and xp.all(xp.linalg.eigvals(mat + mat.T) <= 0)


def cond(linop: Array | LinOp) -> float:
    """Return the condition number κ

    The condition number κ is definied as

    κ = max(λ) / min(λ)

    where λ are eigen values of `linop`.

    Parameters
    ----------
    linop: LinOp or array-like
        An implicit linear operator or a matrix.
    """
    mat = asmatrix(linop)
    xp = arr_api.get_namespace(mat)
    eig = xp.linalg.eigvals(mat)
    return np.max(eig) / np.min(eig)


def fcond(linop: LinOp, tol: float = 0.1) -> float:
    """Estimate the condition number κ

    The condition number κ is definied as

    κ = |max(λ)| / |min(λ)|

    where the two extreme eigen values λ of `linop` are estimated with Lanczos
    algorithm via `scipy.sparse.linalg.eigsh`.

    Parameters
    ----------
    linop: LinOp
        An implicit linear operator.
    tol: float
        The tolerance parameter for `scipy.sparse.linalg.eigsh`.
    """
    try:
        import scipy.sparse.linalg  # ty:ignore[unresolved-import]
    except ImportError as e:
        raise ImportError("scipy is required for fcond") from e
    eig = scipy.sparse.linalg.eigsh(
        scipy.sparse.linalg.aslinearoperator(linop),
        k=2,
        return_eigenvectors=False,
        which="BE",
        tol=tol,
    )
    return np.abs(np.max(eig)) / np.abs(np.min(eig))
