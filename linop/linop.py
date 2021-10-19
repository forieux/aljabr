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

This module implement implicit linear operator. It is wrapper around callables
or functions for ease of use as linear operator and more expressiveness. For
instance, it can wraps the `fft()` function, giving the impression that it is a
matrix. It provides base classes, common operators, and some specialised ones.

"""

import abc
import time
from functools import wraps
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pywt  # type: ignore
import udft
# from icecream import ic  # type: ignore
from numpy import ndarray as array
from numpy.random import standard_normal as randn

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
    "SumOp",
    "asmatrix",
    "dottest",
    "get_broadcast_shape",
    "Identity",
    "Diag",
    "DFT",
    "RealDFT",
    "Conv",
    "Diff",
    "DWT",
    "Analysis",
    "Synthesis",
]

Shape = Tuple[int, ...]
ArrOrSeq = Union[array, Sequence[array]]
ArrOrLinOp = TypeVar("ArrOrLinOp", array, "LinOp")


def _vect(point: ArrOrSeq) -> array:
    if isinstance(point, array):
        return np.reshape(point, (-1, 1))
    return np.concatenate((arr.reshape((-1, 1)) for arr in point), axis=0)


def _unvect(point: array, shapes: Union[Shape, Sequence[Shape]]) -> ArrOrSeq:
    if isinstance(shapes[0], tuple):
        idxs = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
        return [
            np.reshape(point[idxs[i] : idxs[i + 1]], s)  # type: ignore
            for i, s in enumerate(shapes)
        ]
    return np.reshape(point, shapes)  # type: ignore


def _timeit(func):
    """Decorator to time the execution of methods (first argument must be self)"""

    @wraps(func)
    def composite(*args, **kwargs):
        timestamp = time.time()
        out = func(*args, **kwargs)
        duration = time.time() - timestamp

        setattr(args[0], f"duration_{func.__name__}", duration)
        if hasattr(args[0], f"all_duration_{func.__name__}"):
            getattr(args[0], f"all_duration_{func.__name__}").append(duration)
        else:
            setattr(args[0], f"all_duration_{func.__name__}", [duration])

        return out

    # Return our composite function
    return composite


class _TimedMeta(type):
    """MetaClass that adds methods timing"""

    def __new__(cls, clsname, bases, clsdict):
        clsobj = super().__new__(cls, clsname, bases, clsdict)

        for name, value in vars(clsobj).items():
            if callable(value) and name in ("__init__", "forward", "adjoint", "fwadj"):
                setattr(clsobj, name, _timeit(value))

        return clsobj


TimedABCMeta = type("TimedABCMeta", (abc.ABCMeta, _TimedMeta), {})


# class LinOp(metaclass=abc.ABCMeta):
class LinOp(metaclass=TimedABCMeta):
    """Base class for linear operator.

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
        The shape of the operator.
    name : str, optional
        The name of the operator.
    dtype : numpy dtype, optional
        The dtype of the operator.
    H : LinOp
        The Adjoint of the operator.
    """

    def __init__(self, ishape: Shape, oshape: Shape, name: str = "_", dtype=np.float64):
        self.name = name
        self.ishape = ishape
        self.oshape = oshape
        self.dtype = dtype

    @property
    def isize(self):
        """The input size `N = prod(ishape)`."""
        return np.prod(self.ishape)

    @property
    def osize(self):
        """The output size `M = prod(oshape)`."""
        return np.prod(self.oshape)

    @property
    def shape(self):
        """The shape `(M, N)` of the operator."""
        return (self.osize, self.isize)

    @property
    def ndim(self):
        """The number of dimension, always 2."""
        return 2

    @property
    def H(self) -> "LinOp":
        """Return the adjoint `Aᴴ` as a ``LinOp``."""
        return Adjoint(self)

    @abc.abstractmethod
    def forward(self, point: array) -> array:
        """Returns the forward application `A·x`."""
        return NotImplemented

    @abc.abstractmethod
    def adjoint(self, point: array) -> array:
        """Returns the adjoint application `Aᴴ·y`."""
        return NotImplemented

    def matvec(self, point: array) -> array:
        """Vectorized forward application `A·x`.

        Apply `forward` on array of shape (N, 1), returns array of shape (M, 1).
        """
        return np.reshape(self.forward(np.reshape(point, self.ishape)), (-1, 1))

    def rmatvec(self, point: array) -> array:
        """Vectorized adjoint application `Aᴴ·y`.

        Apply `adjoint` on array of shape (M, 1), returns array of shape (N, 1).
        """
        return np.reshape(self.adjoint(np.reshape(point, self.oshape)), (-1, 1))

    def fwadj(self, point: array) -> array:
        """Apply `Aᴴ·A` operator."""
        return self.adjoint(self.forward(point))

    def dot(self, point: array) -> ArrOrSeq:
        """Apply `A` operator"""
        return self.forward(point)

    def hdot(self, point: array) -> array:
        """Apply the adjoint, i.e. `Aᴴ` operator."""
        return self.adjoint(point)

    def __add__(self, obj: "LinOp") -> "LinOp":
        if isinstance(obj, LinOp):
            return SumOp(self, obj)
        raise TypeError("the operand must be a LinOp")

    def __mul__(self, point: ArrOrLinOp) -> ArrOrLinOp:
        if isinstance(point, LinOp):
            return ProdOp(self, point)
        return self.forward(point)

    def __rmul__(self, point: array) -> array:
        return self.adjoint(point)

    def __matmul__(self, point: ArrOrLinOp) -> ArrOrLinOp:
        if isinstance(point, LinOp):
            return ProdOp(self, point)
        return self.matvec(point)

    def __rmatmul__(self, point: array) -> array:
        return self.rmatvec(point)

    def __call__(self, point: array) -> array:
        return self.forward(point)

    def __repr__(self):
        return f"{self.name} ({type(self).__name__}): {self.ishape} → {self.oshape}"


#%%\
class Adjoint(LinOp):
    """The adjoint `Aᴴ` of a linear operator `A`.

    Attributs
    ---------
    base_linop: LinOp
        The base linear operator.
    """

    def __new__(cls, linop: LinOp):
        if isinstance(linop, Adjoint):
            return linop.base_linop
        return super().__new__(cls)

    def __init__(self, linop: LinOp):
        """The adjoint of `linop`.

        If `linop` is alread an `Adjoint` return the `base_linop`."""
        super().__init__(
            linop.oshape, linop.ishape, f"Adjoint of {linop.name}", linop.dtype
        )
        self.base_linop = linop

    def forward(self, point: array) -> array:
        return self.base_linop.adjoint(point)

    def adjoint(self, point: array) -> array:
        return self.base_linop.forward(point)


class Explicit(LinOp):
    """Explicit linear operator from matrix instance."""

    def __init__(self, matrix: array, name="_"):
        """Explicit operator from matrix

        Parameters
        ----------
        mat : array-like
            A 2D array as explicit form of A. `mat` must have `dot`, `transpose`
            and `conj` methods (OK with Numpy).

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
        ishape: Shape,
        oshape: Shape,
        adjoint: Callable[[array], array] = None,
        fwadj: Callable[[array], array] = None,
        name: str = "_",
        dtype=np.complex128,
    ):
        super().__init__(ishape, oshape, name, dtype)
        self._forward = forward
        self._adjoint = adjoint
        self._fwadj = fwadj

    def forward(self, point: array) -> array:
        return self._forward(point)

    def adjoint(self, point: array) -> array:
        if self._adjoint is not None:
            return self._adjoint(point)
        raise NotImplementedError("try to call `adjoint` but not provided.")

    def fwadj(self, point: array) -> array:
        if self._fwadj is not None:
            return self._fwadj(point)
        raise NotImplementedError("try to call `fwadj` but not provided.")


class ProdOp(LinOp):
    """The product of two operators."""

    def __init__(self, left: LinOp, right: LinOp):
        """The sum of two operators.
        Parameters
        ----------
        left: LinOp
            The left operator.
        right: LinOp
            The right operator.
        """
        if left.ishape != right.oshape:
            raise ValueError("`left` input shape must equal `right` output shape")
        super().__init__(right.ishape, left.oshape, name=f"{left.name} × {right.name})")
        self.left = left
        self.right = right

    def forward(self, point: array) -> array:
        return self.left.forward(self.right.forward(point))

    def adjoint(self, point: array) -> array:
        return self.right.adjoint(self.left.adjoint(point))

    def fwadj(self, point: array) -> array:
        return self.right.adjoint(self.left.fwadj(self.right.forward(point)))


class SumOp(LinOp):
    """The sum of two operators."""

    def __init__(self, left: LinOp, right: LinOp):
        """The sum of two operators.

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
            name=f"{left.name} + {right.name})",
            dtype=left.dtype,
        )
        self.left = left
        self.right = right

    def forward(self, point: array) -> array:
        return self.left.forward(point) + self.right.forward(point)

    def adjoint(self, point: array) -> array:
        return self.right.adjoint(point) + self.left.adjoint(point)


#%% \
def asmatrix(linop: LinOp) -> array:
    """Return the matrix corresponding to the linear operator

    Computing the matrix is heavy since it's involve the application of the
    forward callable to `N` unit vectors with `N` the size of the input vector.
    """
    inarray = np.empty((linop.isize, 1))
    matrix = np.empty(linop.shape, dtype=linop.dtype)
    for idx in range(linop.isize):
        inarray.fill(0)
        inarray[idx] = 1
        matrix[:, idx] = linop.matvec(inarray.reshape(linop.ishape))
    return matrix


def dottest(linop: LinOp, num: int = 1, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Dot test.

    Generate random vectors `u` and `v` and perform the dot-test to verify
    the validity of forward and adjoint operators with equality

    `(Aᴴ·u)ᴴ·v = uᴴ·(A·v)`.

    This test can help to detect errors in implementation.

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
    The `u` and `v` vectors passed to `linop` are Numpy arrays.
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


def get_broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> Shape:
    """Return the shape of the broacasted result"""
    shape_a, shape_b = list(shape_a), list(shape_b)
    if len(shape_a) < len(shape_b):
        shape_a = (len(shape_b) - len(shape_a)) * [1] + shape_a
    shape_b = (len(shape_a) - len(shape_b)) * [1] + shape_b
    out_shape: Shape = tuple()
    for sha, shb in zip(shape_a, shape_b):
        if sha == 1 and shb != 1:
            out_shape += (shb,)
        elif shb == 1 and sha != 1:
            out_shape += (sha,)
        elif sha != 1 and shb != 1 and sha != shb:
            raise ValueError("One of the dimension must equal 1 or both must be equal")
    return out_shape


#%% \
class Identity(LinOp):
    """Identity operator of specific shape"""

    def __init__(self, shape: Shape, name: str = "_"):
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

    def __init__(self, diag: array, name: str = "_"):
        """A diagonal operator.

        Parameters
        ----------
        diag : array
            The diagonal of the operator. Input and output must of the same shape.
        """
        super().__init__(diag.shape, diag.shape, name=name)
        self._diag = diag
        self.dtype = diag.dtype

    def forward(self, point: array) -> array:
        return self._diag * point

    def adjoint(self, point: array) -> array:
        return self._diag.conj() * point

    def fwadj(self, point: array) -> array:
        return np.abs(self._diag) ** 2 * point


#%% \
class DFT(LinOp):
    """Discrete Fourier Transform."""

    def __init__(self, shape: Shape, ndim: int, name: str = "DFT"):
        """Unitary discrete Fourier transform.

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


class RealDFT(LinOp):
    """Real Discrete Fourier Transform."""

    def __init__(self, shape: Shape, ndim: int, name: str = "rDFT"):
        """Real unitary discrete Fourier transform.

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
        self._ndim = ndim

    def forward(self, point: array) -> array:
        return udft.rdftn(point, ndim=self._ndim)

    def adjoint(self, point: array) -> array:
        return udft.irdftn(point, self.ishape[-self._ndim :])


class Conv(LinOp):
    """ND convolution on last `N` axis.

    Does not suppose periodic or circular condition.

    Attributes
    ----------
    imp_resp : array
        The impulse response.
    freq_resp : array
        The frequency response.

    Notes
    -----
    Use fft internally for fast computation. The ``forward`` methods is
    equivalent to "valid" boudary condition and ``adjoint`` is equivalent to
    "full" boundary condition with zero filling.

    """

    def __init__(
        self, ir: array, ishape: Shape, dim: int, name: str = "_", dtype=np.float64
    ):
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
            ishape,
            get_broadcast_shape(ishape, ir.shape[:-dim] + ishape[-dim:]),
            name,
            dtype,
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

    def __init__(self, axis: int, ishape: Shape, name: str = "_"):
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
        name: str = "dwt",
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


class Analysis(LinOp):
    """Analysis operator with stationary wavelet decomposition."""

    def __init__(
        self,
        shape: Shape,
        level: int,
        wavelet: str = "haar",
        name: str = "Analysis",
    ):
        """Analysis operator with stationary wavelet decomposition.

        Parameters
        ----------
        shape : tuple of int
            The input shape.
        level : int
            The decomposition level.
        wavelet : str, optional
            The wavelet to use.
        """
        super().__init__(
            shape, (2 * shape[0], 2 * level * shape[1]), name, dtype=np.float64
        )
        self.wlt = wavelet
        self.lvl = level

    def forward(self, point: array) -> array:
        coeffs = pywt.swt2(
            point, wavelet=self.wlt, level=self.lvl, norm=True, trim_approx=True
        )
        return self._coeffs2array(coeffs)

    def adjoint(self, point: array) -> array:
        return pywt.iswt2(self._array2coeffs(point), self.wlt, norm=True)

    def _array2coeffs(self, point: array):
        split = np.split(point, 3 * self.lvl + 1, axis=1)
        clist = [split[0]]
        for lvl in range(self.lvl):
            clist.append([split[3 * lvl + 1], split[3 * lvl + 2], split[3 * lvl + 3]])
        return clist

    @staticmethod
    def _coeffs2array(coeffs) -> array:
        clist = [coeffs[0]]
        for coeff in coeffs[1:]:
            clist.extend([coeff[0], coeff[1], coeff[2]])
        return np.concatenate(clist, axis=1)


class Synthesis(LinOp):
    """Synthesis operator with stationary wavelet decomposition."""

    def __init__(
        self,
        shape: Shape,
        level: int,
        wavelet: str = "haar",
        name: str = "Synthesis",
    ):
        """Analysis operator with stationary wavelet decomposition.

        Parameters
        ----------
        shape : tuple of int
            The input shape.
        level : int
            The decomposition level.
        wavelet : str, optional
            The wavelet to use.
        """
        self.analysis = Analysis(shape, level, wavelet, f"{name} adj.")
        super().__init__(
            self.analysis.oshape, self.analysis.ishape, name, dtype=np.float64
        )
        self.wlt = self.analysis.wlt
        self.lvl = self.analysis.lvl

    def forward(self, point: array) -> array:
        return self.analysis.adjoint(point)

    def adjoint(self, point: array) -> array:
        return self.analysis.forward(point)


# Local Variables:
# ispell-local-dictionary: "english"
# End:
