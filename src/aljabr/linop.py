# Copyright (c) 2013, 2026 F. Orieux <francois.orieux@universite-paris-saclay.fr>

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
wrappers around callables or functions for ease of use as linear operator and
more expressiveness. For instance, it can wraps the `fft()` function, giving the
impression that it is a matrix.

"""

import abc
import math
import time
import warnings
from functools import wraps
from typing import (
    Any,
    Callable,
    Protocol,
    Sequence,
    runtime_checkable,
)

import array_api_compat as arr_api
import numpy as np

Array = Any  # array API standard array — no stable cross-backend type yet
DType = Any  # dtype — no stable cross-backend type yet

__author__ = "François Orieux"
__copyright__ = "2011, 2026, F. Orieux <francois.orieux@universite-paris-saclay.fr>"
__credits__ = ["François Orieux"]
__license__ = "Public domain"
__version__ = "0.4.0"
__maintainer__ = "François Orieux"
__email__ = "francois.orieux@universite-paris-saclay.fr"
__status__ = "beta"
__url__ = "https://github.com/forieux/aljabr"

# __all__ = [
#     "LinOpLike",
#     "LinOp",
#     "Scaled",
#     "Adjoint",
#     "Symmetric",
#     "Explicit",
#     "FuncLinOp",
#     "ProdOp",
#     "AddOp",
#     "SubOp",
#     "asmatrix",
#     "dottest",
#     "fwadjtest",
#     "cond",
#     "fcond",
#     "is_sym",
#     "is_pos_def",
#     "is_semi_pos_def",
#     "is_neg_def",
#     "is_semi_neg_def",
#     "Identity",
#     "Diag",
#     "DFT",
#     "RealDFT",
#     "Conv",
#     "DirectConv",
#     "CircConv",
#     "FreqFilter",
#     "Diff",
#     "DWT",
#     "Analysis2",
#     "Synthesis2",
#     "Slice",
# ]

Shape = tuple[int, ...]


def vectorize(point: Array | Sequence[Array]) -> Array:
    """Vectorize an array or list of arrays as a column vector.

    Parameters
    ----------
    point : Array or list of Array
        A single array or a list of arrays to concatenate.

    Returns
    -------
    Array
        Column vector of shape ``(N, 1)``.
    """
    xp = arr_api.get_namespace(point)
    if isinstance(point, Sequence):
        return xp.concat([xp.reshape(arr, (-1, 1)) for arr in point], axis=0)
    return xp.reshape(point, (-1, 1))


def unvectorize(
    point: Array, shapes: Shape | Sequence[Shape]
) -> Array | Sequence[Array]:
    """Unvectorize a column vector into an array or a list of arrays.

    Parameters
    ----------
    point : Array
        Column vector of shape ``(N, 1)``.
    shapes : Shape or list of Shape
        Target shape, or list of shapes to split into.

    Returns
    -------
    Array or list of Array
        Reshaped array, or list of arrays with the given shapes.
    """
    xp = arr_api.get_namespace(point)
    if isinstance(shapes[0], tuple):
        idxs: list[int] = list(np.cumsum([0] + [int(np.prod(s)) for s in shapes]))
        return [
            xp.reshape(point[idxs[i] : idxs[i + 1]], s) for i, s in enumerate(shapes)
        ]
    return xp.reshape(point, shapes)


@runtime_checkable
class LinOpLike(Protocol):
    """Structural protocol for duck-type LinOp compatibility.

    Any object exposing ``forward``, ``adjoint``, ``fwadj``, ``ishape``,
    ``oshape`` and ``dtype`` satisfies this protocol without inheriting from
    ``LinOp``.  Used in operator overloads to accept external objects.
    """

    ishape: tuple[int, ...]
    oshape: tuple[int, ...]
    dtype: Any

    def forward(self, point: Array) -> Array: ...
    def adjoint(self, point: Array) -> Array: ...
    def fwadj(self, point: Array) -> Array: ...


def timeit(func: Callable) -> Callable:
    """Decorator to time the execution of methods.

    After each call, sets an attribute on ``self`` with the measured duration.
    For ``__init__``, the attribute is ``self.init_last_duration``; for any
    other method named ``name``, it is ``self.name_last_duration``.

    Parameters
    ----------
    func : Callable
        The method to wrap (first argument must be ``self``).

    Returns
    -------
    Callable
        Wrapped method with timing.
    """

    @wraps(func)
    def timed(*args, **kwargs):
        self = args[0]

        timestamp = time.time()
        out = func(*args, **kwargs)
        duration = time.time() - timestamp
        fname: str = func.__name__  # ty:ignore[unresolved-attribute]

        if fname == "__init__":
            setattr(self, "init_last_duration", duration)
        setattr(self, f"{fname}_last_duration", duration)

        return out

    # Return our timed function
    return timed


def checkshape(func: Callable) -> Callable:
    """Decorator to warn about input and output shape mismatches.

    Applies to ``forward``, ``adjoint``, and ``fwadj`` methods. Emits a
    warning if the input or output array shape does not match the shapes
    declared in the ``LinOp`` object (``ishape`` / ``oshape``).

    Parameters
    ----------
    func : Callable
        The method to wrap (first argument must be a ``LinOp`` instance).

    Returns
    -------
    Callable
        Wrapped method with shape checking.
    """

    @wraps(func)
    def shape_checked(self, inarray):
        fname: str = func.__name__  # ty:ignore[unresolved-attribute]

        if fname in ("forward", "fwadj") and inarray.shape != self.ishape:
            warnings.warn(
                f"Input shape {inarray.shape} from `[{type(self)}]{self.name}.{fname}` "
                f"does not equal [{type(self)}]{self.name}.ishape={self.ishape}"
            )
        elif fname in ("adjoint") and inarray.shape != self.oshape:
            warnings.warn(
                f"Input shape {inarray.shape} from `[{type(self)}]{self.name}.{fname}` "
                f"does not equal [{type(self)}]{self.name}.oshape={self.oshape}"
            )

        outarray = func(self, inarray)

        if fname in ("forward") and outarray.shape != self.oshape:
            warnings.warn(
                f"Output shape {outarray.shape} from `{self.name}.{fname}` "
                f"does not equal {self.name}.oshape={self.oshape}"
            )
        elif fname in ("adjoint", "fwadj") and outarray.shape != self.ishape:
            warnings.warn(
                f"Output shape {outarray.shape} from `{self.name}.{fname}` "
                f"does not equal {self.name}.ishape={self.ishape}"
            )

        return outarray

    # Return our shape checked function
    return shape_checked


class LinOp(abc.ABC):
    """An Abstract Base class for linear operator.

    User must implement at least `forward` and `adjoint` methods in their
    concrete class.


    Attributes
    ----------
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
    name : str
        The name of the operator.
    dtype : dtype
        The dtype of the operator (float by default).
    H : LinOp
        The `Adjoint` of the operator `A`.
    S : LinOp
        The `Symmetric` `Aᴴ·A`.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically decorate methods of subclass

        __init__ is timed
        forward, adjoint and fwadj are timed and input and output shape are checked at runtime"""
        for name, value in vars(cls).items():
            if name in ("__init__"):
                setattr(cls, name, timeit(value))
            if name in ("forward", "adjoint", "fwadj"):
                setattr(cls, name, checkshape(timeit(value)))

        super().__init_subclass__(**kwargs)

    def __init__(
        self, ishape: Shape, oshape: Shape, name: str = "·", dtype=float, xp=np
    ):
        """
        Parameters
        ----------
        ishape : tuple of int
            The shape of the input.
        oshape : tuple of int
            The shape of the output.
        name : str, optional
            The name of the operator.
        dtype : dtype, optional
            The dtype of the operator (float by default).
        xp : array namespace, optional
            The array API namespace to use (default: numpy).
        """

        self.name: str = name
        self.ishape: tuple[int, ...] = tuple(ishape)
        self.oshape: tuple[int, ...] = tuple(oshape)
        self.dtype: DType = dtype
        self.xp = xp

    @property
    def isize(self) -> int:
        """The input size `N = np.prod(ishape)`."""
        return math.prod(self.ishape)

    @property
    def osize(self) -> int:
        """The output size `M = np.prod(oshape)`."""
        return math.prod(self.oshape)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape `(self.osize, self.isize)` of the matrix."""
        return (self.osize, self.isize)

    # property to be in read only
    @property
    def ndim(self) -> int:
        """The number of dimension (always 2)."""
        return 2

    @property
    def H(self) -> "LinOp":
        """Return the adjoint `Aᴴ` as a `LinOp`.

        If `A` is already an `Adjoint`, return the original operator."""
        return Adjoint(self)

    @property
    def S(self) -> "LinOp":
        """Return the `Symmetric` `Aᴴ·A`."""
        return Symmetric.from_linop(self)

    @abc.abstractmethod
    def forward(self, point: Array) -> Array:
        """Returns the forward application `A·x`."""
        ...

    @abc.abstractmethod
    def adjoint(self, point: Array) -> Array:
        """Returns the adjoint application `Aᴴ·y`."""
        ...

    def matvec(self, point: Array) -> Array:
        """Vectorized forward application `A·x`.

        Parameters
        ----------
        point : Array
            Column vector of shape ``(N, 1)``.

        Returns
        -------
        Array
            Column vector of shape ``(M, 1)``.
        """
        return self.xp.reshape(
            self.forward(self.xp.reshape(point, self.ishape)), (-1, 1)
        )

    def rmatvec(self, point: Array) -> Array:
        """Vectorized adjoint application `Aᴴ·y`.

        Parameters
        ----------
        point : Array
            Column vector of shape ``(M, 1)``.

        Returns
        -------
        Array
            Column vector of shape ``(N, 1)``.
        """
        return self.xp.reshape(
            self.adjoint(self.xp.reshape(point, self.oshape)), (-1, 1)
        )

    def fwadj(self, point: Array) -> Array:
        """Apply `Aᴴ·A` to `point`.

        Parameters
        ----------
        point : Array
            Input array of shape ``ishape``.

        Returns
        -------
        Array
            Output array of shape ``ishape``.
        """
        return self.adjoint(self.forward(point))

    def asmatrix(self) -> Array:
        """Return the matrix corresponding to the linear operator.

        Applies `forward` to `N` unit vectors where `N = linop.isize`.

        Returns
        -------
        Array
            2D array of shape ``(osize, isize)``.

        Notes
        -----
        Can be very heavy depending on the size of operator.
        """
        inarray = self.xp.empty((self.isize, 1))
        matrix = self.xp.zeros(self.shape, dtype=self.dtype)
        for idx in range(self.isize):
            inarray[idx] = 1
            matrix[:, idx] = self.xp.reshape(self.matvec(inarray), (-1,))
            inarray[idx] = 0
        return matrix

    def __add__(self, value: "LinOp") -> "LinOp":
        """Add (as `+`) a `LinOp` to return an `AddOp`."""
        if isinstance(value, LinOpLike):
            return AddOp(self, value)
        raise TypeError("the operand must be a LinOp")

    def __sub__(self, value: "LinOp") -> "LinOp":
        """Substract (as `-`) a `LinOp` to return an `AddOp`."""
        if isinstance(value, LinOpLike):
            return SubOp(self, value)
        raise TypeError("the operand must be a LinOp")

    def __mul__(self, value: Array | "LinOp") -> Array | "LinOp":
        """Left multiply `*` a LinOp or array

        If `value` is a LinOp duck type, return a ProdOp. Else return `A·x`,
        that is application of `forward(value)`.
        """
        if isinstance(value, LinOpLike):
            return ProdOp(self, value)
        return self.forward(value)

    def __rmul__(self, point: Array) -> Array | "LinOp":
        """Right multiply `*` a scalar or array.

        if `value` is a scalar, return a `Scaled`.

        Otherwise, `value` is considered as an array and return `yᵀ·A`, the
        adjoint application `Aᴴ·y`.
        """
        if isinstance(
            point,
            (int, float, complex),
        ):
            return Scaled(self, point)
        return self.adjoint(point)

    def __matmul__(self, value: Array | "LinOp") -> Array | "LinOp":
        """Left matrix multiply `@` a LinOp or array

        If `value` is a LinOp duck type, return a `ProdOp`.

        If `value is self.H`, return `Symmetric(value)`.

        If `value` is an array, return `matvec(value)`.
        """
        if isinstance(value, LinOpLike):
            if Adjoint(self) is value or self is Adjoint(value):
                return Symmetric.from_linop(value)
            return ProdOp(self, value)
        return self.matvec(value)

    def __rmatmul__(self, point: Array | complex) -> Array | "LinOp":
        """Right matrix multiply `@` a scalar or array.

        if `value` is a scalar, return a `Scaled`.

        Otherwise, `value` is considered as an array and return `yᵀ·A = Aᴴ·y`,
        as `rmatvec(point)`.
        """
        if isinstance(
            point,
            (int, float, complex),
        ):
            return Scaled(self, point)
        return self.rmatvec(point)

    def __call__(self, point: Array) -> Array:
        """Return `A·x` as forward(x)"""
        return self.forward(point)

    def __repr__(self):
        return f"{self.name} ({type(self).__name__}): {self.ishape} → {self.oshape}"


def asmatrix(linop: Array | LinOp, like: Array | None = None) -> Array:
    """Return the matrix corresponding to a linear operator or array.

    Calls `linop.asmatrix()` if `linop` is a `LinOp`. Otherwise converts to
    array using `xp.asarray` (inferred from `like`) or `numpy.asarray`.

    Parameters
    ----------
    linop : Array or LinOp
        The linear operator or array to convert.
    like : Array, optional
        If provided and `linop` is not a `LinOp`, use its array namespace.

    Returns
    -------
    Array
        2D array representation.

    Notes
    -----
    The `LinOp.asmatrix()` method can be very heavy depending on operator size.
    """
    if isinstance(linop, LinOp):
        return linop.asmatrix()
    if like is not None:
        return arr_api.get_namespace(like).asarray(linop)
    return np.asarray(linop)


class BaseOp(LinOp):
    """A `LinOp` defined by callables rather than subclassing."""

    def __init__(
        self,
        forward: Callable[[Array], Array],
        adjoint: Callable[[Array], Array],
        ishape: Shape,
        oshape: Shape,
        fwadj: Callable[[Array], Array] | None = None,
        name: str = "·",
        dtype: DType = float,
        xp=np,
    ):
        """LinOp defined by callables.

        Parameters
        ----------
        forward : callable
            The forward function ``x → A·x``.
        adjoint : callable
            The adjoint function ``y → Aᴴ·y``.
        ishape : tuple of int
            Shape of the input.
        oshape : tuple of int
            Shape of the output.
        fwadj : callable, optional
            The ``Aᴴ·A`` function. Defaults to ``adjoint(forward(x))``.
        name : str, optional
            Name of the operator.
        dtype : dtype, optional
            Dtype of the operator.
        xp : array namespace, optional
            The array API namespace (default: numpy).
        """
        super().__init__(ishape, oshape, name, dtype, xp)
        self.f_forward = forward
        self.f_adjoint = adjoint
        self.f_fwadj = fwadj

    def forward(self, point: Array) -> Array:
        return self.f_forward(point)

    def adjoint(self, point: Array) -> Array:
        return self.f_adjoint(point)

    def fwadj(self, point: Array) -> Array:
        if self.f_fwadj is None:
            return self.f_adjoint(self.f_forward(point))
        return self.f_fwadj(point)


class Scaled(LinOp):
    """An operator `B` scaled by a scalar `γ`.

    Attributes
    ----------
    baseop : LinOp
        The base linear operator `B`.
    scale : float
        The scale factor `γ`.
    """

    def __init__(self, linop: LinOp, scale: complex):
        """An operator `B` scaled by a scalar `γ` (i.e. `A = γ·B`).

        Parameters
        ----------
        linop : LinOp
            The base linear operator `B`.
        scale : float or complex
            The scale factor `γ`.
        """
        self.baseop = linop
        self.scale = scale
        super().__init__(
            linop.ishape, linop.oshape, f"γ{linop.name}", linop.dtype, linop.xp
        )

    def forward(self, point: Array) -> Array:
        return self.scale * self.baseop.forward(point)

    def adjoint(self, point: Array) -> Array:
        return self.xp.conj(self.scale) * self.baseop.adjoint(point)

    def fwadj(self, point: Array) -> Array:
        return self.xp.abs(self.scale) ** 2 * self.baseop.fwadj(point)

    def asmatrix(self, like: Array | None = None):
        return self.scale * asmatrix(self.baseop, like=like)


class Symmetric(LinOp):
    """`A` operator where `Aᴴ = A = Bᴴ·B`.

    >>> Adjoint(A) is A == True

    """

    def __init__(
        self,
        forward: Callable[[Array], Array],
        shape: Shape,
        name="S",
        dtype=float,
        xp=np,
    ):
        """Symmetric operator defined by a callable.

        Parameters
        ----------
        forward : callable
            The function implementing both ``forward`` and ``adjoint``.
        shape : tuple of int
            The (square) shape of the input and output.
        name : str, optional
            Name of the operator.
        dtype : dtype, optional
            Dtype of the operator.
        xp : array namespace, optional
            The array API namespace (default: numpy).
        """
        self._forward = forward

        super().__init__(shape, shape, name, dtype, xp)

    @classmethod
    def from_linop(cls, linop: LinOp):
        """Given `B`, returns `A = Bᴴ·B` (and `Aᴴ = A`)."""
        return cls(
            linop.fwadj,
            linop.ishape,
            f"{linop.name}ᴴ·{linop.name}",
            linop.dtype,
            linop.xp,
        )

    @property
    def H(self) -> "LinOp":
        """Return self: the adjoint of a symmetric operator is itself."""
        return self

    def forward(self, point: Array) -> Array:
        """Returns the application `A·x`."""
        return self._forward(point)

    def adjoint(self, point: Array) -> Array:
        """Returns the adjoint application `Aᴴ·y = A·y`."""
        return self.forward(point)


class Adjoint(LinOp):
    """The adjoint `Aᴴ` of a linear operator `A`.

    `Adjoint` is a singleton: `Adjoint(Adjoint(A)) is A`.

    Delegates to `A` methods.

    Attributes
    ----------
    baseop : LinOp
        The base linear operator.
    """

    def __new__(cls, linop: LinOp):
        # If linop's class overrides H, it knows what its adjoint is — delegate
        # to it instead of wrapping blindly. This comparison checks the property
        # object on the class itself (not the instance), so it is True only when
        # the subclass has actually redefined H (e.g. Symmetric returns self,
        # Adjoint returns baseop). A plain LinOp subclass inherits LinOp.H
        # unchanged, so the condition is False and we fall through to creating a
        # real Adjoint. This keeps Adjoint closed to modification: adding a new
        # "self-knowing" operator only requires overriding H there, not here.
        if type(linop).H is not LinOp.H:
            return linop.H
        return super().__new__(cls)

    def __init__(self, linop: LinOp):
        """Wrap `linop` as its adjoint, or unwrap if `linop` is already an `Adjoint`.

        Parameters
        ----------
        linop : LinOp
            The operator to adjoint.
        """
        # When __new__ returns an existing object, Python still calls __init__
        # on it — we must guard against silently overwriting its attributes.
        # Two cases to bail out early:
        #   - not isinstance(self, Adjoint): __new__ returned a Symmetric or
        #     some other LinOp that is not an Adjoint at all.
        #   - hasattr(self, "baseop"): __new__ returned an already-initialised
        #     Adjoint (e.g. Adjoint(Adjoint(A)) unwraps to A which may itself
        #     be an Adjoint). Re-running __init__ would corrupt it.
        if not isinstance(self, Adjoint) or hasattr(self, "baseop"):
            return

        # ishape/oshape are swapped: the adjoint maps output space → input space.
        super().__init__(
            linop.oshape, linop.ishape, f"{linop.name}ᴴ", linop.dtype, linop.xp
        )
        self.baseop = linop

    @property
    def H(self) -> "LinOp":
        """Return the original operator (adjoint of adjoint is identity)."""
        return self.baseop

    def forward(self, point: Array) -> Array:
        return self.baseop.adjoint(point)

    def adjoint(self, point: Array) -> Array:
        return self.baseop.forward(point)

    def asmatrix(self) -> Array:
        return self.xp.matrix_transpose(self.xp.conj(self.baseop.asmatrix()))


class Explicit(LinOp):
    """Explicit linear operator from matrix instance."""

    def __init__(self, matrix: Array, ishape=None, oshape=None, name="_"):
        """Explicit operator from a 2D matrix.

        Parameters
        ----------
        matrix : Array
            A 2D array representing the operator. The namespace is inferred
            from this array.
        ishape : tuple of int, optional
            Input shape. Defaults to ``(matrix.shape[1], 1)``.
        oshape : tuple of int, optional
            Output shape. Defaults to ``(matrix.shape[0], 1)``.
        name : str, optional
            Name of the operator.
        """
        xp = arr_api.get_namespace(matrix)

        if ishape is None:
            ishape: tuple[int, int] = (matrix.shape[1], 1)
        if oshape is None:
            oshape: tuple[int, int] = (matrix.shape[0], 1)

        if xp.prod(ishape) != matrix.shape[1]:
            raise ValueError("`ishape` must = matrix.shape[1]")
        if xp.prod(oshape) != matrix.shape[0]:
            raise ValueError("`oshape` must = matrix.shape[0]")

        if matrix.ndim != 2:
            raise ValueError("array must have attribute `ndim == 2`")

        self.mat: Array = matrix
        super().__init__(ishape, oshape, name, matrix.dtype, xp)

    def forward(self, point: Array) -> Array:
        return self.xp.reshape(
            self.xp.asarray(self.mat @ self.xp.reshape(point, (-1, 1))),
            self.oshape,
        )

    def adjoint(self, point: Array) -> Array:
        return self.xp.reshape(
            self.xp.asarray(self.xp.conj(self.mat.T) @ point.reshape((-1, 1))),
            self.ishape,
        )

    def asmatrix(self):
        return self.xp.asarray(self.mat)


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
        if left.xp != right.xp:
            warnings.warn("`left` and `right` must have the same array API namespace")
        super().__init__(
            right.ishape,
            left.oshape,
            name=f"({left.name} * {right.name})",
            dtype=left.dtype,
            xp=left.xp,
        )
        self.left = left
        self.right = right

    def forward(self, point: Array) -> Array:
        return self.left.forward(self.right.forward(point))

    def adjoint(self, point: Array) -> Array:
        return self.right.adjoint(self.left.adjoint(point))

    def fwadj(self, point: Array) -> Array:
        return self.right.adjoint(self.left.fwadj(self.right.forward(point)))

    def asmatrix(self):
        return self.xp.matmul(asmatrix(self.left), asmatrix(self.right))


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
        if left.xp != right.xp:
            warnings.warn("`left` and `right` must have the same array API namespace")
        super().__init__(
            left.ishape,
            left.oshape,
            name=f"({left.name} + {right.name})",
            dtype=left.dtype,
            xp=left.xp,
        )
        self.left = left
        self.right = right

    def forward(self, point: Array) -> Array:
        return self.left.forward(point) + self.right.forward(point)

    def adjoint(self, point: Array) -> Array:
        return self.right.adjoint(point) + self.left.adjoint(point)

    def asmatrix(self):
        return asmatrix(self.left) + asmatrix(self.right)


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
        if left.xp != right.xp:
            warnings.warn("`left` and `right` must have the same array API namespace")
        super().__init__(
            left.ishape,
            left.oshape,
            name=f"({left.name} - {right.name})",
            dtype=left.dtype,
            xp=left.xp,
        )
        self.left = left
        self.right = right

    def forward(self, point: Array) -> Array:
        return self.left.forward(point) - self.right.forward(point)

    def adjoint(self, point: Array) -> Array:
        return self.left.adjoint(point) - self.right.adjoint(point)

    def asmatrix(self):
        return asmatrix(self.left) - asmatrix(self.right)


class VStack(LinOp):
    """Vertical stack: maps x → vect([A₀x, A₁x, ...]).

    All operators must share the same ``ishape``. ``forward`` returns a column
    vector of shape ``(sum(op.osize), 1)``. Use ``apply`` to get per-operator
    outputs as a list, or ``split`` to slice a column vector back into
    per-operator shapes.

    ``VStack([A, B, C]).H`` returns ``HStack([Aᴴ, Bᴴ, Cᴴ])``, and
    ``HStack([A, B, C]).H`` returns ``VStack([Aᴴ, Bᴴ, Cᴴ])``.

    So if y = vect([y₀, y₁, ...]) of shape (N, 1), and A = VStack([A₀, A₁,
    ...]), then y = Ax can be obtained with

    >>> y_list = A.apply(x)          # list of per-operator outputs
    >>> y = A.forward(x)             # stacked column vector, shape A.oshape
    >>> A.split(y) == y_list         # split back to per-operator shapes

    Notes
    -----
    This operator is for convenience. The recommendation is to write a custom
    operator that directly inherits from `LinOp` and implements `forward` and
    `adjoint`.

    """

    def __init__(self, oplist: Sequence[LinOp], name="[·]"):
        """Vertical stack of operators.

        Parameters
        ----------
        oplist : sequence of LinOp
            Operators to stack. All must share the same ``ishape``.
        name : str, optional
            Name of the operator.
        """
        if len({op.ishape for op in oplist}) > 1:
            raise ValueError("all operators must have the same ishape")
        if len({id(op.xp) for op in oplist}) > 1:
            warnings.warn("operators have different array API namespaces")

        self.oplist: list[LinOp] = list(oplist)

        osizes = [math.prod(op.oshape) for op in oplist]
        self._oshapes: list[Shape] = [op.oshape for op in oplist]

        self._hstack: "HStack | None" = None

        super().__init__(
            oplist[0].ishape,
            (sum(osizes), 1),
            name=name,
            dtype=oplist[0].dtype,
            xp=oplist[0].xp,
        )

    @property
    def H(self) -> "HStack":
        """Return `HStack([Aᴴ, Bᴴ, ...])`, cached."""
        if self._hstack is None:
            self._hstack = HStack([Adjoint(op) for op in self.oplist])
        return self._hstack

    def apply(self, point: Array) -> Array | list[Array]:
        """Apply each operator and return results as a list preserving shapes."""
        return [op.forward(point) for op in self.oplist]

    def forward(self, point: Array) -> Array:
        return vectorize(self.apply(point))

    def adjoint(self, point: Array) -> Array:
        arrays = self.split(point)
        result = self.oplist[0].adjoint(arrays[0])
        for op, arr in zip(self.oplist[1:], arrays[1:]):
            # +=, iadd, not possible with array-api standard arrays
            result = result + op.adjoint(arr)
        return result

    def split(self, point: Array) -> Array | list[Array]:
        """Split the output column vector back into per-operator shaped arrays."""
        return unvectorize(point, self._oshapes)


class HStack(LinOp):
    """Horizontal stack: maps vect([x₀, x₁, ...]) → Σ opᵢ(xᵢ).

    Dual of ``VStack``: all operators must share the same ``oshape``.
    ``forward`` splits the input column vector by each operator's ``ishape``,
    applies ``op.forward``, and sums. ``adjoint`` applies each ``op.adjoint``
    to the same input and vectorizes the results.

    ``VStack([A, B, C]).H`` returns ``HStack([Aᴴ, Bᴴ, Cᴴ])``, and
    ``HStack([A, B, C]).H`` returns ``VStack([Aᴴ, Bᴴ, Cᴴ])``.

    So if x = vect([x₀, x₁, ...]) of shape (M, 1), and A = HStack([A₀, A₁,
    ...]), then y = Ax can be obtained with

    >>> x_list = A.split(x)          # list of per-operator inputs
    >>> y = A.forward(x)             # sum of op.forward(xᵢ), shape A.oshape
    >>> A.apply(x) == x_list         # apply each op to its sub-input

    Notes
    -----
    This operator is for convenience. The recommendation is to write a custom
    operator that directly inherits from `LinOp` and implements `forward` and
    `adjoint`.

    """

    def __init__(self, oplist: Sequence[LinOp], name="[·|·]"):
        """Horizontal stack of operators.

        Parameters
        ----------
        oplist : sequence of LinOp
            Operators to stack. All must share the same ``oshape``.
        name : str, optional
            Name of the operator.
        """
        if len({op.oshape for op in oplist}) > 1:
            raise ValueError("all operators must have the same oshape")
        if len({id(op.xp) for op in oplist}) > 1:
            warnings.warn("operators have different array API namespaces")

        self.oplist: list[LinOp] = list(oplist)

        isizes = [math.prod(op.ishape) for op in oplist]
        self._ishapes: list[Shape] = [op.ishape for op in oplist]

        self._vstack: "VStack | None" = None

        super().__init__(
            (sum(isizes), 1),
            oplist[0].oshape,
            name=name,
            dtype=oplist[0].dtype,
            xp=oplist[0].xp,
        )

    @property
    def H(self) -> "VStack":
        """Return `VStack([Aᴴ, Bᴴ, ...])`, cached."""
        if self._vstack is None:
            self._vstack = VStack([Adjoint(op) for op in self.oplist])
        return self._vstack

    def forward(self, point: Array) -> Array:
        arrays = self.split(point)
        result = self.oplist[0].forward(arrays[0])
        for op, arr in zip(self.oplist[1:], arrays[1:]):
            # +=, iadd, not possible with array-api standard arrays
            result = result + op.forward(arr)
        return result

    def adjoint(self, point: Array) -> Array:
        return vectorize([op.adjoint(point) for op in self.oplist])

    def apply(self, point: Array) -> list[Array]:
        """Split the input and apply each operator, returning results as a list."""
        return [op.forward(arr) for op, arr in zip(self.oplist, self.split(point))]

    def split(self, point: Array) -> Array | list[Array]:
        """Split the input column vector into per-operator shaped arrays."""
        return unvectorize(point, self._ishapes)


# Local Variables:
# ispell-local-dictionary: "english"
# End:
