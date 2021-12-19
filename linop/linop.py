# Copyright (c) 2013, 2021 F. Orieux <francois.orieux@universite-paris-saclay.fr>

# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or distribute
# this software, either in source code form or as a compiled binary, for any
# purpose, commercial or non-commercial, and by any means.
#
# In jurisdictions that recognize copyright laws, the author or authors of this
# software dedicate any and all copyright interest in the software to the public
# domain. We make this dedication for the benefit of the public at large and to
# the detriment of our heirs and successors. We intend this dedication to be an
# overt act of relinquishment in perpetuity of all present and future rights to
# this software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <https://unlicense.org>


"""The ``linop`` module
====================

This module implements an interface for implicit linear operator. It is mostly
an empty shell, a wrapper around callables or functions for ease of use as
linear operator and more expressiveness. For instance, it can wraps the `fft()`
function, giving the impression that it is a matrix. It provides base classes,
common concrete operators, and some specialised ones.

"""

import abc
import time
import warnings
from functools import wraps
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np  # type: ignore
import pywt  # type: ignore
import scipy.signal  # type: ignore
import udft
# from icecream import ic  # type: ignore
from numpy import ndarray as array  # type: ignore
from numpy.random import standard_normal as randn  # type: ignore

__author__ = "François Orieux"
__copyright__ = "2011, 2021, F. Orieux <francois.orieux@universite-paris-saclay.fr>"
__credits__ = ["François Orieux"]
__license__ = "Public domain"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "francois.orieux@universite-paris-saclay.fr"
__status__ = "beta"
__url__ = "https://https://github.com/forieux/linearop"

__all__ = [
    "LinOp",
    "Adjoint",
    "Explicit",
    "FuncLinOp",
    "ProdOp",
    "AddOp",
    "SubOp",
    "asmatrix",
    "dottest",
    "Identity",
    "Diag",
    "DFT",
    "RealDFT",
    "Conv",
    "DirectConv",
    "CircConv",
    "FreqFilter",
    "Diff",
    "DWT",
    "Analysis2",
    "Synthesis2",
]

Shape = Tuple[int, ...]
ArrOrSeq = Union[array, Sequence[array]]
ArrOrLinOp = TypeVar("ArrOrLinOp", array, "LinOp")


def vectorize(point: ArrOrSeq) -> array:
    """Vectorize an array or list of array as column vector"""
    if isinstance(point, array):
        return np.reshape(point, (-1, 1))
    return np.concatenate((arr.reshape((-1, 1)) for arr in point), axis=0)


def unvectorize(point: array, shapes: Union[Shape, Sequence[Shape]]) -> ArrOrSeq:
    """Unvectorize a column vector as an array or list of array"""
    if isinstance(shapes[0], tuple):
        idxs = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
        return [
            np.reshape(point[idxs[i] : idxs[i + 1]], s) for i, s in enumerate(shapes)
        ]
    return np.reshape(point, shapes)


def timeit(func):
    """Decorator to time the execution of methods (first argument must be `self`)"""

    @wraps(func)
    def timed(*args, **kwargs):
        self = args[0]

        timestamp = time.time()
        out = func(*args, **kwargs)
        duration = time.time() - timestamp

        setattr(self, f"{func.__name__}_last_duration", duration)
        if (
            hasattr(self, f"all_{func.__name__}_duration")
            and func.__name__ != "__init__"
        ):
            getattr(self, f"all_{func.__name__}_duration").append(duration)
        else:
            setattr(self, f"all_{func.__name__}_duration", [duration])

        return out

    # Return our timed function
    return timed


def checkshape(func):
    """Decorator to warn about input and output shape of forward, ajoint and fwadj."""

    @wraps(func)
    def shape_checked(self, inarray):
        if func.__name__ in ("forward", "fwadj") and inarray.shape != self.ishape:
            warnings.warn(
                f"Input shape {inarray.shape} from `{self.name}.{func.__name__}` "
                f"does not equal {self.name}.ishape={self.ishape}"
            )
        elif func.__name__ in ("adjoint") and inarray.shape != self.oshape:
            warnings.warn(
                f"Input shape {inarray.shape} from `{self.name}.{func.__name__}` "
                f"does not equal {self.name}.oshape={self.oshape}"
            )

        outarray = func(self, inarray)

        if func.__name__ in ("forward") and outarray.shape != self.oshape:
            warnings.warn(
                f"Output shape {outarray.shape} from `{self.name}.{func.__name__}` "
                f"does not equal {self.name}.oshape={self.oshape}"
            )
        elif func.__name__ in ("adjoint", "fwadj") and outarray.shape != self.ishape:
            warnings.warn(
                f"Output shape {outarray.shape} from `{self.name}.{func.__name__}` "
                f"does not equal {self.name}.ishape={self.ishape}"
            )

        return outarray

    # Return our shape checked function
    return shape_checked


class TimedMeta(type):
    """MetaClass that adds methods timing and shape checking."""

    def __new__(cls, clsname, bases, clsdict):
        clsobj = super().__new__(cls, clsname, bases, clsdict)

        for name, value in vars(clsobj).items():
            if name in ("__init__"):
                setattr(clsobj, name, timeit(value))
            if name in ("forward", "adjoint", "fwadj"):
                setattr(clsobj, name, checkshape(timeit(value)))

        return clsobj


TimedABCMeta = type("TimedABCMeta", (TimedMeta, abc.ABCMeta), {})


def is_linop_duck(obj):
    if (
        hasattr(obj, "forward")
        and callable(obj.forward)
        and hasattr(obj, "adjoint")
        and callable(obj.adjoint)
        and hasattr(obj, "fwadj")
        and callable(obj.fwadj)
        and hasattr(obj, "ishape")
        and hasattr(obj, "oshape")
        and hasattr(obj, "dtype")
    ):
        return True
    return False


class LinOp(metaclass=TimedABCMeta):
    """An Abstract Base class for linear operator.

    User must implement at least `forward` and `adjoint` methods in their
    concrete class.


    Attributs
    ---------
    ishape : tuple of int
        The shape of the input.
    oshape : tuple of int
        The shape of the output.
    isize : int
        The input size.
    osize : int
        The output size.
    shape : tuple of two int.
        The shape of the operator as matrix.
    name : str, optional
        The name of the operator.
    dtype : numpy dtype, optional
        The dtype of the operator (np.float64 by default).
    H : LinOp
        The Adjoint of the operator.

    """

    def __init__(self, ishape: Shape, oshape: Shape, name: str = "_", dtype=np.float64):
        """
        Parameters
        ---------
        ishape : tuple of int
            The shape of the input.
        oshape : tuple of int
            The shape of the output.
        name : str, optional
            The name of the operator.
        dtype : numpy dtype, optional
            The dtype of the operator (np.float64 by default).
        """

        self.name = name
        self.ishape = tuple(ishape)
        self.oshape = tuple(oshape)
        self.dtype = dtype

    @property
    def isize(self):
        """The input size `N = np.prod(ishape)`."""
        return np.prod(self.ishape)

    @property
    def osize(self):
        """The output size `M = np.prod(oshape)`."""
        return np.prod(self.oshape)

    @property
    def shape(self):
        """The shape `(self.osize, self.isize)` of the matrix."""
        return (self.osize, self.isize)

    @property
    def ndim(self):
        """The number of dimension (always 2)."""
        return 2

    @property
    def H(self) -> "LinOp":  # pylint: disable=invalid-name
        """Return the adjoint `Aᴴ` as a `LinOp`.

        I A is already and Adjoint, return the original operator."""
        return Adjoint(self)

    @abc.abstractmethod
    def forward(self, point: array) -> array:
        """Returns the forward application `A·x`."""
        raise NotImplementedError

    def adjoint(self, point: array) -> array:
        """Returns the adjoint application `Aᴴ·y`."""
        raise NotImplementedError

    def matvec(self, point: array) -> array:
        """Vectorized forward application `A·x`.

        Apply `forward` on array of shape (N, 1), returns array of shape (M, 1).
        The reshape are done internally.

        """
        return np.reshape(self.forward(np.reshape(point, self.ishape)), (-1, 1))

    def rmatvec(self, point: array) -> array:
        """Vectorized adjoint application `Aᴴ·y`.

        Apply `adjoint` on array of shape (M, 1), returns array of shape (N, 1).
        The reshape are done internally

        """
        return np.reshape(self.adjoint(np.reshape(point, self.oshape)), (-1, 1))

    def fwadj(self, point: array) -> array:
        """Apply `Aᴴ·A` operator."""
        return self.adjoint(self.forward(point))

    def dot(self, point: array) -> ArrOrSeq:
        """Returns the forward application `A·x`."""
        return self.forward(point)

    def hdot(self, point: array) -> array:
        """Returns the adjoint application `Aᴴ·y`."""
        return self.adjoint(point)

    def __add__(self, value: "LinOp") -> "LinOp":
        """Add (as `+`) a `LinOp` to return a `AddOp`."""
        if is_linop_duck(value):
            return AddOp(self, value)
        raise TypeError("the operand must be a LinOp")

    def __sub__(self, value: "LinOp") -> "LinOp":
        """Substract (as `-`) a `LinOp` to return a `AddOp`."""
        if is_linop_duck(value):
            return SubOp(self, value)
        raise TypeError("the operand must be a LinOp")

    def __mul__(self, value: ArrOrLinOp) -> ArrOrLinOp:
        """Multiply `*` a LinOp or array

        if `value` is a LinOp duck type, return a ProdOp. Else return `A·x`,
        that is application of `forward(value)`.

        """
        if is_linop_duck(value):
            return ProdOp(self, value)
        return self.forward(value)

    def __rmul__(self, point: array) -> array:
        """Return `yᵀ·A`, the adjoint application `Aᴴ·y`."""
        return self.adjoint(point)

    def __matmul__(self, value: ArrOrLinOp) -> ArrOrLinOp:
        """Matrix multiply `@` a LinOp or array

        If `value` is a LinOp duck type, return a ProdOp.

        If `Adjoint(self) is value`, return `Symmetric(value)`.

        If `value` is an array, return `matvec(value)`.

        """
        if is_linop_duck(value):
            if Adjoint(self) is value:
                return Symmetric(value)
            return ProdOp(self, value)
        return self.matvec(value)

    def __rmatmul__(self, point: array) -> array:
        """Return x·Aᴴ as rmatvec(point)"""
        return self.rmatvec(point)

    def __call__(self, point: array) -> array:
        """Return `A·x` as forward(x)"""
        return self.forward(point)

    def __repr__(self):
        return f"{self.name} ({type(self).__name__}): {self.ishape} → {self.oshape}"


#%%\
class Symmetric(LinOp):
    """A operator where `Aᴴ = A = Bᴴ·B`.

    >>> Adjoint(A) is A == True

    Attributs
    ---------
    orig_linop: LinOp
        The base linear operator `B`.
    """

    def __init__(self, linop: LinOp):
        """A operator where `Aᴴ = A = Bᴴ·B`.

        Parameters
        ----------
        linop: LinOp
            The base linear operator `B`.
        """
        self.orig_linop = linop
        super().__init__(
            linop.ishape, linop.ishape, f"{linop.name}ᴴ·{linop.name}", linop.dtype
        )

    def forward(self, point: array) -> array:
        """Returns the application `A·x`."""
        return self.orig_linop.fwadj(point)

    def adjoint(self, point: array) -> array:
        """Returns the adjoint application `Aᴴ·y = A·y`."""
        return self.forward(point)

    def __getattr__(self, name):
        try:
            return getattr(self.orig_linop, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Original LinOp of `Symmetric` has no {name} attribut"
            ) from exc


class Adjoint(LinOp):
    """The adjoint `Aᴴ` of a linear operator `A`.

    `Adjoint` are singleton

    >>> Adjoint(Adjoint(A)) is A == True

    Attributs
    ---------
    orig_linop: LinOp
        The base linear operator.
    """

    def __new__(cls, linop: LinOp):
        if isinstance(linop, Symmetric):
            return linop
        if isinstance(linop, Adjoint):
            return linop.orig_linop
        return super().__new__(cls)

    def __init__(self, linop: LinOp):
        """The adjoint of `linop`.

        If `linop` is alread an `Adjoint` return the `orig_linop`."""
        super().__init__(linop.oshape, linop.ishape, f"{linop.name}ᴴ", linop.dtype)
        self.orig_linop = linop

    def forward(self, point: array) -> array:
        return self.orig_linop.adjoint(point)

    def adjoint(self, point: array) -> array:
        return self.orig_linop.forward(point)

    def __getattr__(self, name):
        try:
            return getattr(self.orig_linop, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Original LinOp of `Adjoint` has no {name} attribut"
            ) from exc


class Explicit(LinOp):
    """Explicit linear operator from matrix instance."""

    def __init__(self, matrix: array, name="_"):
        """Explicit operator from matrix

        Parameters
        ----------
        mat : array-like
            A 2D array as explicit form of A. `mat` must have `dot`, `transpose`
            and `conj` methods (available with Numpy).

        Notes
        -----
        The `forward` and `adjoint` input array are reshaped as column vector
        before `dot` call.
        """

        super().__init__((matrix.shape[1], 1), (matrix.shape[0], 1), name, matrix.dtype)
        if matrix.ndim != 2:
            raise ValueError("array must have attribut `ndim == 2`")
        self.mat = matrix

    def forward(self, point: array) -> array:
        return np.asanyarray(self.mat.dot(point.reshape((-1, 1))))

    def adjoint(self, point: array) -> array:
        return np.asanyarray(self.mat.transpose().conj().dot(point.reshape((-1, 1))))


class FuncLinOp(LinOp):
    """A linear operator `LinOp` defined with callables."""

    def __init__(
        self,
        forward: Callable[[array], array],
        adjoint: Callable[[array], array],
        ishape: Shape,
        oshape: Shape,
        fwadj: Callable[[array], array] = None,
        name: str = "_",
        dtype=np.float64,
    ):
        super().__init__(ishape, oshape, name, dtype)
        self.f_forward = forward
        self.f_adjoint = adjoint
        self.f_fwadj = fwadj

    def forward(self, point: array) -> array:
        return self.f_forward(point)

    def adjoint(self, point: array) -> array:
        return self.f_adjoint(point)

    def fwadj(self, point: array) -> array:
        if self.f_fwadj is None:
            return self.f_adjoint(self.f_forward(point))
        return self.f_fwadj(point)


class ProdOp(LinOp):
    """The product of two operators `A·B`."""

    def __init__(self, left: LinOp, right: LinOp):
        """The product of two operators `A·B`.
        Parameters
        ----------
        left: LinOp
            The left operator `A`.
        right: LinOp
            The right operator `B`.
        """
        if left.ishape != right.oshape:
            warnings.warn("`left` input shape must equal `right` output shape")
        super().__init__(
            right.ishape, left.oshape, name=f"({left.name} * {right.name})"
        )
        self.left = left
        self.right = right

    def forward(self, point: array) -> array:
        return self.left.forward(self.right.forward(point))

    def adjoint(self, point: array) -> array:
        return self.right.adjoint(self.left.adjoint(point))

    def fwadj(self, point: array) -> array:
        return self.right.adjoint(self.left.fwadj(self.right.forward(point)))


class AddOp(LinOp):
    """The sum of two operators `A + B`."""

    def __init__(self, left: LinOp, right: LinOp):
        """The sum of two operators `A + B`.

        Parameters
        ----------
        left: LinOp
            The left operator.
        right: LinOp
            The right operator.
        """
        if (left.ishape != right.ishape) or (left.oshape != right.oshape):
            raise ValueError("operators must have the same input and output shape")
        super().__init__(
            left.ishape,
            left.oshape,
            name=f"({left.name} + {right.name})",
            dtype=left.dtype,
        )
        self.left = left
        self.right = right

    def forward(self, point: array) -> array:
        return self.left.forward(point) + self.right.forward(point)

    def adjoint(self, point: array) -> array:
        return self.right.adjoint(point) + self.left.adjoint(point)


class SubOp(LinOp):
    """The substraction of two operators `A - B`."""

    def __init__(self, left: LinOp, right: LinOp):
        """The substraction of two operators `A - B`.

        Parameters
        ----------
        left: LinOp
            The left operator.
        right: LinOp
            The right operator.
        """
        if (left.ishape != right.ishape) or (left.oshape != right.oshape):
            raise ValueError("operators must have the same input and output shape")
        super().__init__(
            left.ishape,
            left.oshape,
            name=f"({left.name} - {right.name})",
            dtype=left.dtype,
        )
        self.left = left
        self.right = right

    def forward(self, point: array) -> array:
        return self.left.forward(point) - self.right.forward(point)

    def adjoint(self, point: array) -> array:
        return self.right.adjoint(point) - self.left.adjoint(point)


#%% \
def asmatrix(linop: LinOp) -> array:
    """Return the matrix corresponding to the linear operator

    Parameters
    ----------
    linop: LinOp
        The linear operator to represent as matrix

    Notes
    -----

    Computing the matrix can heavy since it's involve the application of the
    `linop.forward` to `N` unit vectors with `N = linop.iszie` is the size of
    the input.

    """
    inarray = np.empty((linop.isize, 1))
    matrix = np.empty(linop.shape, dtype=linop.dtype)
    for idx in range(linop.isize):
        inarray.fill(0)
        inarray[idx] = 1
        matrix[:, idx] = linop.matvec(inarray).ravel()
    return matrix


def dottest(linop: LinOp, num: int = 1, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
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

    Notes
    -----

    The `u` and `v` vectors passed to `linop` are 1D random float Numpy arrays
    and the function use the `matvec` and `rmatvec` methods of `LinOp`.

    """
    test = True
    for _ in range(num):
        uvec = randn(linop.isize)
        vvec = randn(linop.osize)
        test = test & np.allclose(
            linop.matvec(uvec).conj() * vvec,
            uvec.conj() * linop.rmatvec(vvec),
            rtol=rtol,
            atol=atol,
        )
    return test


#%% \
class Identity(LinOp):
    """Identity operator of specific shape

    Notes
    -----
    The `forward` and `adjoint` apply `np.asarray` on their input.

    """

    def __init__(self, shape: Shape, name: str = "I"):
        """The identity operator.

        Parameters
        ----------
        shape : tuple of int
            The shape of the identity.


        """
        super().__init__(shape, shape, name=name, dtype=np.float64)

    def forward(self, point: array) -> array:
        return np.asarray(point)

    def adjoint(self, point: array) -> array:
        return np.asarray(point)


class Diag(LinOp):
    """Diagonal operator."""

    def __init__(self, diag: array, name: str = "D"):
        """A diagonal operator.

        Parameters
        ----------
        diag : array
            The diagonal of the operator. Input and output have the same shape.

        """
        super().__init__(diag.shape, diag.shape, name=name, dtype=diag.dtype)
        self.diag = diag

    def forward(self, point: array) -> array:
        return self.diag * point

    def adjoint(self, point: array) -> array:
        return np.conj(self.diag) * point

    def fwadj(self, point: array) -> array:
        return np.abs(self.diag) ** 2 * point


#%% \
class DFT(LinOp):
    """Discrete Fourier Transform on the N last axis."""

    def __init__(self, shape: Shape, ndim: int, name: str = "DFT"):
        """Unitary discrete Fourier transform on the N last axis.

        Parameters
        ----------
        shape : tuple of int
            The shape of the input
        ndim : int
            The number of last axes over which to compute the DFT.
        """
        super().__init__(shape, shape, name=name, dtype=np.complex128)
        self.dim = ndim

    def forward(self, point: array) -> array:
        return udft.dftn(point, ndim=self.dim)

    def adjoint(self, point: array) -> array:
        return udft.idftn(point, ndim=self.dim)

    def fwadj(self, point: array) -> array:
        return array


class RealDFT(LinOp):
    """Real Discrete Fourier Transform on the N last axis."""

    def __init__(self, shape: Shape, ndim: int, name: str = "rDFT"):
        """Real unitary discrete Fourier transform on the N last axis.

        Parameters
        ----------
        shape : tuple of int
            The shape of the input
        ndim : int
            The number of last axes over which to compute the DFT.
        """
        super().__init__(
            shape, shape[:-1] + (shape[-1] // 2 + 1,), name=name, dtype=np.complex128
        )
        self.dim = ndim

    def forward(self, point: array) -> array:
        return udft.rdftn(point, ndim=self.dim)

    def adjoint(self, point: array) -> array:
        return udft.irdftn(point, self.ishape[-self.dim :])

    def fwadj(self, point: array) -> array:
        return array


class Conv(LinOp):
    """ND convolution on last `N` axis.

    Does not suppose periodic or circular condition.

    Attributes
    ----------
    imp_resp : array
        The impulse response.
    freq_resp : array
        The frequency response.
    dim : int
        The last `dim` axis where convolution apply.

    Notes
    -----
    Use fft internally for fast computation. The `forward` methods is equivalent
    to "valid" boudary condition and `adjoint` is equivalent to "full" boundary
    condition with zero filling.

    """

    def __init__(self, ir: array, ishape: Shape, dim: int, name: str = "Conv"):
        """ND convolution on last `N` axis.

        Parameters
        ----------
        ir : array
            The impulse responses. Must be at least of `ndim==dim`.
        ishape : tuple of int
            The shape of the input images. Images are on the last two axis.
        dim : int
            The last `dim` axis where convolution apply.

        """
        super().__init__(
            ishape=ishape,
            oshape=np.broadcast_shapes(ishape, ir.shape[:-dim] + ishape[-dim:]),
            name=name,
        )

        self.dim = dim
        self.imp_resp = ir
        self.freq_resp = udft.ir2fr(ir, self.ishape[-dim:])

        self.margins = ir.shape[-dim:]
        self._slices = [slice(None) for _ in range(len(ishape) - dim)]
        for idx in reversed(range(dim)):
            self._slices.append(
                slice(
                    ir.shape[idx] // 2,
                    ishape[idx] - ir.shape[idx] // 2 + ir.shape[idx] % 2,
                )
            )

    def _dft(self, point: array) -> array:
        return udft.rdftn(point, self.dim)

    def _idft(self, point: array) -> array:
        return udft.irdftn(point, self.ishape[-self.dim :])

    def forward(self, point: array) -> array:
        return self._idft(self._dft(point) * self.freq_resp)[self._slices]

    def adjoint(self, point: array) -> array:
        out = np.zeros(self.ishape)
        out[self._slices] = point
        return self._idft(self._dft(out) * self.freq_resp.conj())

    def fwadj(self, point: array) -> array:
        out = np.zeros_like(point)
        out[self._slices] = self._idft(self._dft(point) * self.freq_resp)[self._slices]
        return self._idft(self._dft(out) * self.freq_resp.conj())


class DirectConv(LinOp):
    """Direct convolution

    The convolution is performed on the last N axis where N = id.ndim.

    Attributes
    ----------
    ir : array
        The impulse response.

    Notes
    -----
    Use Overlap-Add method from `scipy.signal.oaconvolve` if available or
    `convolve` otherwise. Convolution are performed on the last axes.

    """

    def __init__(self, ir: array, ishape: Shape, name: str = "DConv"):
        """Direct convolution

        Parameters
        ----------
        ir : array
            The impulse response.
        ishape: tuple of int
            The shape of the input array.

        """
        oshape = tuple(
            ishape[idx]
            if idx < len(ishape) - len(ir.shape)
            else ishape[idx] - ir.shape[idx - (len(ishape) - len(ir.shape))] + 1
            for idx in range(len(ishape))
        )
        super().__init__(
            ishape=ishape,
            oshape=oshape,
            name=name,
        )
        self.ir = ir  # pylint: disable=invalid-name

    @property
    def ir(self):  # pylint: disable=invalid-name
        """The impulse response"""
        return np.squeeze(self._ir)

    @property
    def freq_resp(self):
        return udft.ir2fr(self.ir, self.ishape)

    @ir.setter
    def ir(self, ir: array):  # pylint: disable=invalid-name
        # Keep internaly an _ir with (1, ) prepend for convolution on last N
        # axis since scipy.signal.convolve wants array with same ndim.
        self._ir = np.reshape(ir, (len(self.ishape) - ir.ndim) * (1,) + ir.shape)

    def forward(self, point: array) -> array:
        if hasattr(scipy.signal, "oaconvolve"):
            return scipy.signal.oaconvolve(point, self._ir, mode="valid")
        return scipy.signal.convolve(point, self._ir, mode="valid")

    def adjoint(self, point: array) -> array:
        if hasattr(scipy.signal, "oaconvolve"):
            return scipy.signal.oaconvolve(point, np.flip(self._ir), mode="full")
        return scipy.signal.convolve(point, np.flip(self._ir), mode="full")


class FreqFilter(Diag):
    """Frequency filter in Fourier space

    Attributes
    ----------
    freq_resp: array
        The frequency response of the filter

    Notes
    -----
    Almost like diagonal but suppose complex Fourier space"""

    def __init__(self, ir: array, ishape: Shape, name: str = "Filter"):
        super().__init__(udft.ir2fr(ir, ishape), name=name)


class CircConv(LinOp):
    """Circulant convolution"""

    def __init__(self, imp_resp: array, shape: Shape, name: str = "CConv"):
        super().__init__(ishape=shape, oshape=shape, name=name)
        self.imp_resp = imp_resp
        self.ffilter = Diag(udft.ir2fr(imp_resp, shape))

    @property
    def freq_resp(self):
        """The frequency response"""
        return self.ffilter.diag

    def _dft(self, arr):
        return udft.rdftn(arr, len(self.ishape))

    def _idft(self, arr):
        return udft.irdftn(arr, self.oshape)

    def forward(self, point: array) -> array:
        return self._idft(self.ffilter.forward(self._dft(point)))

    def adjoint(self, point: array) -> array:
        return self._idft(self.ffilter.adjoint(self._dft(point)))

    def fwadj(self, point: array) -> array:
        return self._idft(self.ffilter.fwadj(self._dft(point)))


class Diff(LinOp):
    """Difference operator.

    Compute the first-order differences along an axis.

    Attributes
    ----------
    axis: int
        The axis along which the differences is performed.

    Notes
    -----
    Use `numpy.diff` and implement the correct adjoint, with `numpy.diff` also.

    """

    def __init__(self, axis: int, ishape: Shape, name: str = "Diff"):
        """First-order differences operator.

        Parameters
        ----------
        axis : int
            The axis along which to perform the diff.

        ishape : tuple of int
            The shape of the input array.
        """
        oshape = list(ishape)
        oshape[axis] = ishape[axis] - 1
        super().__init__(ishape, tuple(oshape), name=name)
        self.axis = axis

    def forward(self, point: array) -> array:
        """The forward application `A·x`.

        This corresponds to the application of the following matrix in 1D.

        -1  1  0  0
         0 -1  1  0
         0  0 -1  1
        """
        return np.diff(point, axis=self.axis)

    def adjoint(self, point: array) -> array:
        """The adjoint application `Aᴴ·y`.

        This corresponds to the application of the following matrix in 1D

        -1  0  0
         1 -1  0
         0  1 -1
         0  0  1

        """
        return -np.diff(point, prepend=0, append=0, axis=self.axis)


class DWT(LinOp):
    """Unitary Discrete Wavelet Transform.

    Attributs
    ---------
    wlt : str
        The wavelet.
    lvl : int
        The decomposition level.
    """

    def __init__(
        self,
        shape: Shape,
        level: Optional[int] = None,
        wavelet: str = "haar",
        name: str = "DWT",
    ):
        """Unitary Discrete Wavelet Transform.

        Parameters
        ----------
        shape : tuple of int
            The input shape.
        level : int, optional
            The decomposition level.
        wavelet : str, optional
            The wavelet to use.
        """
        super().__init__(shape, shape, name, dtype=np.float64)
        self.wlt = wavelet
        self.lvl = level
        self._mode = "periodization"
        self._slices = pywt.coeffs_to_array(
            pywt.wavedecn(
                np.empty(shape), wavelet=wavelet, mode="periodization", level=level
            )
        )[1]

    def forward(self, point: array) -> array:
        return pywt.coeffs_to_array(
            pywt.wavedecn(point, wavelet=self.wlt, mode=self._mode, level=self.lvl)
        )[0]

    def adjoint(self, point: array) -> array:
        return pywt.waverecn(
            pywt.array_to_coeffs(point, self._slices),
            wavelet=self.wlt,
            mode=self._mode,
        )

    def fwadj(self, point: array) -> array:
        return array


class Analysis2(LinOp):
    """2D analysis operator with stationary wavelet decomposition."""

    def __init__(
        self,
        shape: Tuple[int, int],
        level: int,
        wavelet: str = "haar",
        name: str = "A",
    ):
        """2D analysis operator with stationary wavelet decomposition.

        Parameters
        ----------
        shape : tuple of (int, int)
            The input shape.
        level : int
            The decomposition level.
        wavelet : str, optional
            The wavelet to use.
        """
        super().__init__(shape, (3 * level + 1,) + shape, name, dtype=np.float64)
        self.wlt = wavelet
        self.lvl = level
        self.norm = True

    def forward(self, point: array) -> array:
        coeffs = pywt.swt2(
            point, wavelet=self.wlt, level=self.lvl, norm=self.norm, trim_approx=True
        )
        return self.coeffs2cube(coeffs)

    def adjoint(self, point: array) -> array:
        return pywt.iswt2(self.cube2coeffs(point), self.wlt, norm=self.norm)

    def cube2coeffs(self, point: array) -> array:
        """Return pywt coefficients from 3D array"""
        split = np.split(point, 3 * self.lvl + 1, axis=0)
        coeffs_list = [np.squeeze(split[0])]
        for lvl in range(self.lvl):
            coeffs_list.append(
                [
                    np.squeeze(split[3 * lvl + 1]),
                    np.squeeze(split[3 * lvl + 2]),
                    np.squeeze(split[3 * lvl + 3]),
                ]
            )
        return coeffs_list

    @staticmethod
    def coeffs2cube(coeffs) -> array:
        """Return 3D array from pywt coefficients"""
        clist = [coeffs[0][np.newaxis, ...]]
        for coeff in coeffs[1:]:
            clist.extend(
                [
                    coeff[0][np.newaxis, ...],
                    coeff[1][np.newaxis, ...],
                    coeff[2][np.newaxis, ...],
                ]
            )
        return np.concatenate(clist, axis=0)

    def im2coeffs(self, point: array):
        """Return pywt coefficients from an image array"""
        split = np.split(point, 3 * self.lvl + 1, axis=1)
        coeffs_list = [split[0]]
        for lvl in range(self.lvl):
            coeffs_list.append(
                [split[3 * lvl + 1], split[3 * lvl + 2], split[3 * lvl + 3]]
            )
        return coeffs_list

    @staticmethod
    def coeffs2im(coeffs) -> array:
        """Return an image array from pywt coefficients"""
        clist = [coeffs[0]]
        for coeff in coeffs[1:]:
            clist.extend([coeff[0], coeff[1], coeff[2]])
        return np.concatenate(clist, axis=1)

    def cube2im(self, cube):
        return self.coeffs2im(self.cube2coeffs(cube))

    def im2cube(self, im):
        return self.coeffs2cube(self.im2coeffs(im))

    def get_irs(self):
        """Return the impulse response of the filter bank"""
        iarr = np.zeros(self.ishape)
        iarr[0, 0] = 1
        return self.forward(iarr)

    def get_frs(self):
        """Return the frequency response of the filter bank"""
        return np.ascontiguousarray(np.fft.rfftn(self.get_irs(), self.ishape[-2:]))


class Synthesis2(LinOp):
    """2D synthesis operator with stationary wavelet decomposition."""

    def __init__(
        self,
        shape: Tuple[int, int],
        level: int,
        wavelet: str = "haar",
        name: str = "S",
    ):
        """2D synthesis operator with stationary wavelet decomposition.

        Parameters
        ----------
        shape : tuple of (int, int)
            The input shape.
        level : int
            The decomposition level.
        wavelet : str, optional
            The wavelet to use.
        """
        self.analysis = Analysis2(shape, level, wavelet)
        super().__init__(self.analysis.oshape, self.analysis.ishape, name)
        self.wlt = self.analysis.wlt
        self.lvl = self.analysis.lvl

    def forward(self, point: array) -> array:
        return self.analysis.adjoint(point)

    def adjoint(self, point: array) -> array:
        return self.analysis.forward(point)

    def cube2coeffs(self, point: array) -> array:
        """Return pywt coefficients from 3D array"""
        return self.analysis.cube2coeffs(point)

    def coeffs2cube(self, coeffs) -> array:
        """Return 3D array from pywt coefficients"""
        return self.analysis.coeffs2cube(coeffs)

    def im2coeffs(self, point: array):
        """Return pywt coefficients from image"""
        return self.analysis.im2coeffs(point)

    def coeffs2im(self, coeffs) -> array:
        """Return image from pywt coefficients"""
        return self.analysis.coeffs2im(coeffs)

    def cube2im(self, cube):
        return self.analysis.cube2im(cube)

    def im2cube(self, im):
        return self.analysis.im2cube(im)

    def get_irs(self):
        """Rerturn the impulse response of the filter bank."""
        return np.flip(self.analysis.get_irs(), axis=(1, 2))

    def get_frs(self):
        """Rerturn the frequency response of the filter bank."""
        return np.ascontiguousarray(np.fft.rfftn(self.get_irs(), self.ishape[-2:]))


# Local Variables:
# ispell-local-dictionary: "english"
# End:
