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


"""The ``linop`` module
====================

This module implements an interface for implicit linear operators. It is mostly
wrappers around callables or functions for ease of use as linear operators and
more expressiveness. For instance, it can wrap the `fft()` function, giving the
impression that it is a matrix.

"""

from __future__ import annotations

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

type Array = Any  # array API standard array — no stable cross-backend type yet


__all__ = [
    "Shape",
    "Array",
    "vectorize",
    "unvectorize",
    "asmatrix",
    "LinOp",
    "BaseOp",
    "Scaled",
    "Adjoint",
    "Symmetric",
    "Dense",
    "ProdOp",
    "AddOp",
    "SubOp",
    "VStack",
    "HStack",
]

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
    if isinstance(point, Sequence):
        xp = arr_api.get_namespace(*point)
        return xp.concat([xp.reshape(arr, (-1, 1)) for arr in point], axis=0)
    xp = arr_api.get_namespace(point)
    return xp.reshape(point, (-1, 1))


def unvectorize(point: Array, shapes: Sequence[Shape]) -> list[Array]:
    """Unvectorize a column vector into a list of arrays.

    Parameters
    ----------
    point : Array
        Column vector of shape ``(N, 1)``.
    shapes : list of Shape
        List of target shapes to split into.

    Returns
    -------
    list of Array
        List of arrays with the given shapes.
    """
    xp = arr_api.get_namespace(point)
    idxs: list[int] = np.cumsum([0] + [int(np.prod(s)) for s in shapes]).tolist()
    return [xp.reshape(point[idxs[i] : idxs[i + 1]], s) for i, s in enumerate(shapes)]


@runtime_checkable
class LinOpLike(Protocol):
    """Structural protocol for duck-type LinOp compatibility.

    Any object exposing ``forward``, ``adjoint``, ``fwadj``, ``ishape``,
    ``oshape`` satisfies this protocol without inheriting from ``LinOp``. Used
    in operator overloads to accept external objects.

    """

    ishape: tuple[int, ...]
    oshape: tuple[int, ...]

    def forward(self, point: Array) -> Array: ...
    def adjoint(self, point: Array) -> Array: ...
    def fwadj(self, point: Array) -> Array: ...


def timeit(func: Callable) -> Callable:
    """Decorator to time the execution of methods.

    After each call, updates ``self.metadata`` with the measured duration. For
    ``__init__``, sets ``metadata["init"]``; for any other method named
    ``name``, sets ``metadata["name"]``.

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

        # For __init__, record whether metadata existed and was unset before
        # the call.  If metadata["init"] is already set, __init__ is running
        # on an existing object (Adjoint.__new__ returning an existing op) and
        # we must not overwrite the object's real construction time.
        init_was_unset = (
            not hasattr(self, "metadata") or self.metadata.get("init") is None
        )

        timestamp = time.time()
        out = func(*args, **kwargs)
        duration = time.time() - timestamp
        fname: str = func.__name__  # ty: ignore[unresolved-attribute]

        if fname == "__init__":
            if init_was_unset:
                self.metadata["init"] = duration
        else:
            if fname in ("forward", "adjoint", "fwadj"):
                self.metadata[f"{fname}"].append(duration)

        return out

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
        fname: str = func.__name__  # ty: ignore[unresolved-attribute]

        if fname in ("forward", "fwadj") and inarray.shape != self.ishape:
            warnings.warn(
                f"Input shape {inarray.shape} from `[{type(self)}]{self.name}.{fname}` "
                f"does not equal [{type(self)}]{self.name}.ishape={self.ishape}"
            )
        elif fname == "adjoint" and inarray.shape != self.oshape:
            warnings.warn(
                f"Input shape {inarray.shape} from `[{type(self)}]{self.name}.{fname}` "
                f"does not equal [{type(self)}]{self.name}.oshape={self.oshape}"
            )

        outarray = func(self, inarray)

        if fname == "forward" and outarray.shape != self.oshape:
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

    return shape_checked


class LinOp(abc.ABC):
    """An abstract base class for linear operators.

    User must implement at least `forward` and `adjoint` methods in their
    concrete class.

    Parameters
    ----------
    ishape : tuple of int
        The shape of the input.
    oshape : tuple of int
        The shape of the output.
    name : str, optional
        The name of the operator.

    Attributes
    ----------
    metadata : dict
        Timing information populated automatically after each method call. See
        the guide for details.
    """

    # let numpy defer to __rmul__ instead of broadcasting element-wise
    __array_ufunc__ = None

    def __init_subclass__(cls, **kwargs):
        """Automatically decorate methods of subclasses.

        ``__init__`` is timed. ``forward``, ``adjoint``, and ``fwadj`` are
        timed and have their input/output shapes checked at runtime.
        """
        for name, value in vars(cls).items():
            if name == "__init__":
                setattr(cls, name, timeit(value))
            if name in ("forward", "adjoint", "fwadj"):
                setattr(cls, name, checkshape(timeit(value)))

        super().__init_subclass__(**kwargs)

    def __init__(self, ishape: Shape, oshape: Shape, name: str = "·"):
        self.name: str = name
        self.ishape: tuple[int, ...] = tuple(ishape)
        self.oshape: tuple[int, ...] = tuple(oshape)

        self.metadata: dict = {
            "init": None,
            "forward": [],
            "adjoint": [],
            "fwadj": [],
        }

    @property
    def isize(self) -> int:
        """The input size `N = math.prod(ishape)`."""
        return math.prod(self.ishape)

    @property
    def osize(self) -> int:
        """The output size `M = math.prod(oshape)`."""
        return math.prod(self.oshape)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape `(self.osize, self.isize)` of the matrix."""
        return (self.osize, self.isize)

    @property
    def ndim(self) -> int:
        """The number of dimensions (always 2)."""
        return 2

    @property
    def H(self) -> "LinOp":
        """Return `Adjoint(self)`."""
        return Adjoint(self)

    @property
    def G(self) -> "LinOp":
        """Return the Gram operator `Aᴴ·A` as a `Symmetric`."""
        return Symmetric.gram(self)

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
        xp = arr_api.get_namespace(point)
        return xp.reshape(self.forward(xp.reshape(point, self.ishape)), (-1, 1))

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
        xp = arr_api.get_namespace(point)
        return xp.reshape(self.adjoint(xp.reshape(point, self.oshape)), (-1, 1))

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

    def asmatrix(self, like: Array | None = None) -> Array:
        """Return the matrix corresponding to the linear operator.

        Applies `forward` to `N` unit vectors where `N = linop.isize`.

        Parameters
        ----------
        like : Array, optional
            If provided, use its array namespace; otherwise use float64 numpy
            array. See guide.

        Returns
        -------
        Array
            2D array of shape ``(osize, isize)``.

        Notes
        -----
        Can be very heavy depending on the size of the operator.

        """
        xp = arr_api.get_namespace(like) if like is not None else np
        inarray = xp.zeros((self.isize, 1))
        matrix = xp.zeros(
            self.shape, dtype=like.dtype if like is not None else np.float64
        )
        for idx in range(self.isize):
            if arr_api.is_jax_array(inarray):
                inarray = inarray.at[idx].set(1)
            else:
                inarray[idx] = 1
            col = xp.reshape(self.matvec(inarray), (-1,))
            if arr_api.is_jax_array(matrix):
                matrix = matrix.at[:, idx].set(col)
            else:
                matrix[:, idx] = col
            if arr_api.is_jax_array(inarray):
                inarray = inarray.at[idx].set(0)
            else:
                inarray[idx] = 0
        return matrix

    def __add__(self, value: LinOp) -> LinOp:
        """Add (as `+`) a `LinOp` to return an `AddOp`."""
        if isinstance(value, LinOpLike):
            return AddOp(self, value)
        raise TypeError("the operand must be a linear operator")

    def __sub__(self, value: LinOp) -> LinOp:
        """Subtract (as `-`) a `LinOp` to return a `SubOp`."""
        if isinstance(value, LinOpLike):
            return SubOp(self, value)
        raise TypeError("the operand must be a linear operator")

    def __mul__(self, value: Array | LinOp) -> Array | LinOp:
        """Left multiply `*` a LinOp or array.

        If `value` is a LinOp duck type, return a ProdOp. Else return `A·x`,
        that is, the application of `forward(value)`.
        """
        if isinstance(value, LinOpLike):
            return ProdOp(self, value)
        return self.forward(value)

    def __rmul__(self, point: Array) -> Array | LinOp:
        """Right multiply `*` a scalar or array.

        If `point` is a scalar, return a `Scaled`.

        Otherwise, `point` is treated as an array and returns `Aᴴ·y`, the
        adjoint application.
        """
        if isinstance(
            point,
            (int, float, complex),
        ):
            return Scaled(self, point)
        return self.adjoint(point)

    def __matmul__(self, value: Array | LinOp) -> Array | LinOp:
        """Left matrix multiply `@` a LinOp or array.

        If `value` is a LinOp duck type, return a `ProdOp`.

        If `value` is the adjoint of `self` (or vice versa), return the
        Gram operator via `Symmetric.gram`.

        If `value` is an array, return `matvec(value)`.
        """
        if isinstance(value, LinOpLike):
            # Adjoint.__new__ unwraps double adjoints: Adjoint(Adjoint(A)) is A.
            # So Adjoint(self) is value is True when self=Adjoint(A) and
            # value=A, i.e. self @ value = Aᴴ·A. Symmetrically for self is
            # Adjoint(value).
            if Adjoint(self) is value or self is Adjoint(value):
                return Symmetric.gram(value)
            return ProdOp(self, value)
        return self.matvec(value)

    def __rmatmul__(self, point: Array | complex) -> Array | LinOp:
        """Right matrix multiply `@` a scalar or array.

        If `point` is a scalar, return a `Scaled`.

        Otherwise, `point` is treated as a column vector and returns `Aᴴ·y`
        via `rmatvec(point)`.
        """
        if isinstance(
            point,
            (int, float, complex),
        ):
            return Scaled(self, point)
        return self.rmatvec(point)

    def __call__(self, point: Array) -> Array:
        """Return `forward(x)`."""
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
        return linop.asmatrix(like=like)
    if like is not None:
        return arr_api.get_namespace(like).asarray(linop)
    return np.asarray(linop)


class BaseOp(LinOp):
    """A `LinOp` defined by callables rather than subclassing.

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
    """

    def __init__(
        self,
        forward: Callable[[Array], Array],
        adjoint: Callable[[Array], Array],
        ishape: Shape,
        oshape: Shape,
        fwadj: Callable[[Array], Array] | None = None,
        name: str = "·",
    ):
        super().__init__(ishape, oshape, name=name)
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
    """An operator `B` scaled by a scalar `γ` (i.e. `A = γ·B`).

    Parameters
    ----------
    baseop : LinOp
        The base linear operator `B`.
    scale : float or complex
        The scale factor `γ`.

    Attributes
    ----------
    baseop : LinOp
        The base linear operator `B`.
    scale : complex or float
        The scale factor `γ`.
    """

    def __init__(self, baseop: LinOp, scale: complex | float):
        self.baseop = baseop
        self.scale = scale
        super().__init__(
            baseop.ishape,
            baseop.oshape,
            name=f"γ{baseop.name}",
        )

    def forward(self, point: Array) -> Array:
        return self.scale * self.baseop.forward(point)

    def adjoint(self, point: Array) -> Array:
        return self.scale.conjugate() * self.baseop.adjoint(point)

    def fwadj(self, point: Array) -> Array:
        return abs(self.scale) ** 2 * self.baseop.fwadj(point)

    def asmatrix(self, like: Array | None = None) -> Array:
        return self.scale * asmatrix(self.baseop, like=like)


class Symmetric(LinOp):
    """`A` operator where `Aᴴ = A`.

    For any `Symmetric` instance `A`, ``Adjoint(A) is A`` is ``True``:
    it is fully defined by `forward` since `adjoint` delegates to it.

    Parameters
    ----------
    forward : callable
        The function implementing both ``forward`` and ``adjoint``.
    shape : tuple of int
        The (square) shape of the input and output.
    name : str, optional
        Name of the operator.

    """

    def __init__(
        self,
        forward: Callable[[Array], Array],
        shape: Shape,
        name: str = "S",
    ):
        self._forward = forward

        super().__init__(shape, shape, name=name)

    @classmethod
    def gram(cls, linop: LinOp) -> "Symmetric":
        """Given `B`, return the Gram operator `Bᴴ·B` (self-adjoint)."""
        if isinstance(linop, Adjoint):
            name = f"{linop.baseop.name}·{linop.name}"
        else:
            name = f"{linop.name}ᴴ·{linop.name}"
        return cls(
            linop.fwadj,
            linop.ishape,
            name=name,
        )

    @property
    def H(self) -> LinOp:
        """Return self: the adjoint of a symmetric operator is itself."""
        return self

    def forward(self, point: Array) -> Array:
        """Returns the application `A·x`."""
        return self._forward(point)

    def adjoint(self, point: Array) -> Array:
        """Returns the adjoint application `Aᴴ·y = A·y`."""
        return self._forward(point)


class Adjoint(LinOp):
    """The adjoint `Aᴴ` of a linear operator `A`.

    `Adjoint` is an involution: `Adjoint(Adjoint(A)) is A`.

    Delegates to `A` methods.

    Parameters
    ----------
    linop : LinOp
        The operator to adjoint.

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
            linop.oshape,
            linop.ishape,
            name=f"{linop.name}ᴴ",
        )
        self.baseop = linop

    @property
    def H(self) -> LinOp:
        """Return the original operator (adjoint of adjoint is identity)."""
        return self.baseop

    def forward(self, point: Array) -> Array:
        return self.baseop.adjoint(point)

    def adjoint(self, point: Array) -> Array:
        return self.baseop.forward(point)

    def asmatrix(self, like: Array | None = None) -> Array:
        mat = self.baseop.asmatrix(like=like)
        xp = arr_api.get_namespace(mat)
        return xp.matrix_transpose(xp.conj(mat))


class Dense(LinOp):
    """Dense linear operator from matrix instance.

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

    def __init__(
        self,
        matrix: Array,
        ishape: Shape | None = None,
        oshape: Shape | None = None,
        name: str = "_",
    ):
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2-dimensional")

        if ishape is None:
            ishape = (matrix.shape[1], 1)
        if oshape is None:
            oshape = (matrix.shape[0], 1)

        if math.prod(ishape) != matrix.shape[1]:
            raise ValueError("`ishape` must = matrix.shape[1]")
        if math.prod(oshape) != matrix.shape[0]:
            raise ValueError("`oshape` must = matrix.shape[0]")

        self.mat: Array = matrix
        super().__init__(ishape, oshape, name=name)

    def forward(self, point: Array) -> Array:
        xp = arr_api.get_namespace(self.mat)
        return xp.reshape(
            xp.asarray(self.mat @ xp.reshape(point, (-1, 1))), self.oshape
        )

    def adjoint(self, point: Array) -> Array:
        xp = arr_api.get_namespace(self.mat)
        return xp.reshape(
            xp.asarray(
                xp.conj(xp.matrix_transpose(self.mat)) @ xp.reshape(point, (-1, 1))
            ),
            self.ishape,
        )

    def asmatrix(self, like: Array | None = None) -> Array:
        if like is None:
            xp = arr_api.get_namespace(self.mat)
        else:
            xp = arr_api.get_namespace(like)
        return xp.asarray(self.mat)


class ProdOp(LinOp):
    """The product of two operators `A·B`.

    Parameters
    ----------
    left : LinOp
        The left operator `A`.
    right : LinOp
        The right operator `B`.
    """

    def __init__(self, left: LinOp, right: LinOp):
        if left.ishape != right.oshape:
            raise ValueError("`left.ishape` must equal `right.oshape`")
        super().__init__(
            right.ishape,
            left.oshape,
            name=f"({left.name}·{right.name})",
        )
        self.left = left
        self.right = right

    def forward(self, point: Array) -> Array:
        return self.left.forward(self.right.forward(point))

    def adjoint(self, point: Array) -> Array:
        return self.right.adjoint(self.left.adjoint(point))

    def fwadj(self, point: Array) -> Array:
        return self.right.adjoint(self.left.fwadj(self.right.forward(point)))

    def asmatrix(self, like: Array | None = None) -> Array:
        left_mat = asmatrix(self.left, like=like)
        right_mat = asmatrix(self.right, like=like)
        xp = arr_api.get_namespace(left_mat)
        return xp.matmul(left_mat, right_mat)


class AddOp(LinOp):
    """The sum of two operators `A + B`.

    Parameters
    ----------
    left : LinOp
        The left operator.
    right : LinOp
        The right operator.
    """

    def __init__(self, left: LinOp, right: LinOp):
        if (left.ishape != right.ishape) or (left.oshape != right.oshape):
            raise ValueError("operators must have the same input and output shape")
        super().__init__(
            left.ishape,
            left.oshape,
            name=f"({left.name} + {right.name})",
        )
        self.left = left
        self.right = right

    def forward(self, point: Array) -> Array:
        return self.left.forward(point) + self.right.forward(point)

    def adjoint(self, point: Array) -> Array:
        return self.right.adjoint(point) + self.left.adjoint(point)

    def asmatrix(self, like: Array | None = None) -> Array:
        return asmatrix(self.left, like=like) + asmatrix(self.right, like=like)


class SubOp(LinOp):
    """The subtraction of two operators `A - B`.

    Parameters
    ----------
    left : LinOp
        The left operator.
    right : LinOp
        The right operator.
    """

    def __init__(self, left: LinOp, right: LinOp):
        if (left.ishape != right.ishape) or (left.oshape != right.oshape):
            raise ValueError("operators must have the same input and output shape")
        super().__init__(
            left.ishape,
            left.oshape,
            name=f"({left.name} - {right.name})",
        )
        self.left = left
        self.right = right

    def forward(self, point: Array) -> Array:
        return self.left.forward(point) - self.right.forward(point)

    def adjoint(self, point: Array) -> Array:
        return self.left.adjoint(point) - self.right.adjoint(point)

    def asmatrix(self, like: Array | None = None) -> Array:
        return asmatrix(self.left, like=like) - asmatrix(self.right, like=like)


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
    >>> A.split(y)                   # per-operator shapes, same as y_list

    Parameters
    ----------
    oplist : sequence of LinOp
        Operators to stack. All must share the same ``ishape``.
    name : str, optional
        Name of the operator.

    """

    def __init__(self, oplist: Sequence[LinOp], name: str = "[·]"):
        if not oplist:
            raise ValueError("oplist must not be empty")
        if len({op.ishape for op in oplist}) > 1:
            raise ValueError("all operators must have the same ishape")

        self.oplist: list[LinOp] = list(oplist)

        osizes = [math.prod(op.oshape) for op in oplist]
        self._oshapes: list[Shape] = [op.oshape for op in oplist]

        self._hstack: "HStack | None" = None

        super().__init__(
            oplist[0].ishape,
            (sum(osizes), 1),
            name=name,
        )

    @property
    def H(self) -> "HStack":
        """Return `HStack([Aᴴ, Bᴴ, ...])`, cached."""
        if self._hstack is None:
            self._hstack = HStack([Adjoint(op) for op in self.oplist])
        return self._hstack

    def apply(self, point: Array) -> list[Array]:
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

    def split(self, point: Array) -> list[Array]:
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
    >>> A.apply(x)                   # per-operator outputs, same as x_list

    Parameters
    ----------
    oplist : sequence of LinOp
        Operators to stack. All must share the same ``oshape``.
    name : str, optional
        Name of the operator.

    """

    def __init__(self, oplist: Sequence[LinOp], name: str = "[·|·]"):
        if not oplist:
            raise ValueError("oplist must not be empty")
        if len({op.oshape for op in oplist}) > 1:
            raise ValueError("all operators must have the same oshape")

        self.oplist: list[LinOp] = list(oplist)

        isizes = [math.prod(op.ishape) for op in oplist]
        self._ishapes: list[Shape] = [op.ishape for op in oplist]

        self._vstack: "VStack | None" = None

        super().__init__(
            (sum(isizes), 1),
            oplist[0].oshape,
            name=name,
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

    def split(self, point: Array) -> list[Array]:
        """Split the input column vector into per-operator shaped arrays."""
        return unvectorize(point, self._ishapes)


# Local Variables:
# ispell-local-dictionary: "english"
# End:
