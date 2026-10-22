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

"""The ``concrete`` module
=======================

Concrete implementations of common linear operators: convolutions (FFT-based,
direct, circulant), discrete transforms (DFT, DWT, wavelet analysis/synthesis),
diagonal, difference, sampling operators, ...

"""

import types

import array_api_compat as arr_api
import numpy as np
import udft

pywt: types.ModuleType | None
spsignal: types.ModuleType | None

try:
    import pywt
except ImportError:
    pywt = None

try:
    import scipy.signal as spsignal
except ImportError:
    spsignal = None

from .linop import Array, LinOp, Shape

__all__ = [
    "DFT",
    "DWT",
    "Analysis2",
    "CircConv",
    "Conv",
    "Diag",
    "Diff",
    "DirectConv",
    "FreqFilter",
    "Identity",
    "RealDFT",
    "Sampling",
    "Slice",
    "Synthesis2",
]


class Identity(LinOp):
    """Identity operator of specific shape.

    Parameters
    ----------
    shape : tuple of int
        The shape of the input and output.
    name : str, optional
        Name of the operator.
    """

    def __init__(self, shape: Shape, name: str = "I"):
        super().__init__(shape, shape, name=name)

    def forward(self, point: Array) -> Array:
        return point

    def adjoint(self, point: Array) -> Array:
        return point

    def asmatrix(self, like: Array | None = None) -> Array:
        if like is None:
            return np.eye(self.isize)
        xp = arr_api.get_namespace(like)
        return xp.eye(self.isize, dtype=like.dtype, device=arr_api.device(like))


class Diag(LinOp):
    """Diagonal operator.

    Parameters
    ----------
    diag : Array
        The diagonal values. Input and output have the same shape as `diag`.
        The array namespace is inferred from this array.
    name : str, optional
        Name of the operator.
    """

    def __init__(self, diag: Array, name: str = "D"):
        xp = arr_api.get_namespace(diag)
        self.diag = xp.asarray(diag)
        super().__init__(
            self.diag.shape,
            self.diag.shape,
            name=name,
        )

    def forward(self, point: Array) -> Array:
        return self.diag * point

    def adjoint(self, point: Array) -> Array:
        xp = arr_api.get_namespace(self.diag)
        return xp.conj(self.diag) * point

    def fwadj(self, point: Array) -> Array:
        xp = arr_api.get_namespace(self.diag)
        return xp.abs(self.diag) ** 2 * point

    def asmatrix(self, like: Array | None = None) -> Array:
        if like is None:
            xp = arr_api.get_namespace(self.diag)
            diag = self.diag
        else:
            xp = arr_api.get_namespace(like)
            diag = xp.asarray(self.diag, device=arr_api.device(like))
        # `diag` is not in the array API standard: scale the identity columns
        eye = xp.eye(self.isize, dtype=diag.dtype, device=arr_api.device(diag))
        return eye * xp.reshape(diag, (1, -1))


class DFT(LinOp):
    """Discrete Fourier Transform on the last N axes.

    Parameters
    ----------
    shape : tuple of int
        The shape of the input.
    ndim : int
        The number of last axes over which to compute the DFT.
    name : str, optional
        Name of the operator.
    """

    def __init__(self, shape: Shape, ndim: int, name: str = "DFT"):
        self._udft = udft

        super().__init__(shape, shape, name=name)
        self.dim = ndim

    def forward(self, point: Array) -> Array:
        return self._udft.dftn(point, ndim=self.dim)

    def adjoint(self, point: Array) -> Array:
        return self._udft.idftn(point, ndim=self.dim)

    def fwadj(self, point: Array) -> Array:
        return point


class RealDFT(LinOp):
    r"""Real Discrete Fourier Transform on the last N axes.

    This is the unitary `DFT` of a real input, computed only on the
    non-redundant half of the spectrum: the last axis has length `n // 2 + 1`,
    the other half being given by the Hermitian symmetry. It is not a separate
    operator with its own geometry, it is `DFT` restricted to real signals.

    Parameters
    ----------
    shape : tuple of int
        The shape of the input.
    ndim : int
        The number of last axes over which to compute the DFT.
    name : str, optional
        Name of the operator.

    Notes
    -----
    **Exception to the adjoint convention.** Consistently with `DFT`,
    `adjoint` is the inverse transform (`irdftn`) and `fwadj` is the identity.
    This is the adjoint of `DFT` on the full, Hermitian-symmetric, spectrum.
    On the half spectrum, it corresponds to the weighted inner product

    .. math::

        \langle u, v \rangle_w = \Re \sum_k w_k \bar{u}_k v_k,

    where `w_k = 1` on the last axis for the frequency 0 (and `n / 2` if `n`
    is even), and `w_k = 2` otherwise, since these frequencies stand for
    themselves and their conjugate. Then `⟨Aᴴu, x⟩ = ⟨u, A·x⟩_w` and
    `‖A·x‖_w = ‖x‖` (Parseval).

    It is **not** the adjoint for the standard inner product of the half
    spectrum: `‖A·x‖ ≠ ‖x‖` and `dottest` fails. Use `RealDFT` as a change of
    representation, e.g. `Aᴴ·Λ·A` with a diagonal `Λ` from `udft.ir2fr` as in
    `CircConv`, not inside generic algorithms relying on the standard adjoint.
    """

    def __init__(self, shape: Shape, ndim: int, name: str = "rDFT"):
        self._udft = udft

        shape = tuple(shape)  # tuple concatenation below, `shape` may be a list
        super().__init__(shape, shape[:-1] + (shape[-1] // 2 + 1,), name=name)
        self.dim = ndim

    def forward(self, point: Array) -> Array:
        return self._udft.rdftn(point, ndim=self.dim)

    def adjoint(self, point: Array) -> Array:
        return self._udft.irdftn(point, self.ishape[-self.dim :])

    def fwadj(self, point: Array) -> Array:
        return point


class Conv(LinOp):
    """ND convolution on the last `N` axes.

    Does not assume a periodic or circular boundary condition.

    Parameters
    ----------
    ir : Array
        The real impulse response, on its last `dim` axes. Leading axes, if
        any, are batch axes broadcast against the batch axes of the input. The
        namespace is inferred from this array.
    ishape : tuple of int
        The shape of the input. Images are on the last `dim` axes.
    dim : int
        Number of last axes over which convolution applies.
    name : str, optional
        Name of the operator.

    Attributes
    ----------
    imp_resp : Array
        The impulse response.
    freq_resp : Array
        The frequency response.
    dim : int
        The last `dim` axes where convolution applies.

    Raises
    ------
    ValueError
        If `dim` is not in ``[1, len(ishape)]``, if `ir` has less than `dim`
        axes, if its batch axes do not broadcast to the input ones, or if it is
        larger than the input on a convolved axis. If `ir` is complex (from
        `udft.ir2fr`).

    Notes
    -----
    Uses FFT internally for fast computation. The `forward` method is equivalent
    to "valid" boundary condition and `adjoint` is equivalent to "full" boundary
    condition with zero filling.

    Uses the array namespace of the impulse response.

    The operator is real, on the half-spectrum: `ir` and the input must be
    real. For a complex `ir`, use `DirectConv`.

    """

    def __init__(self, ir: Array, ishape: Shape, dim: int, name: str = "Conv"):
        self._udft = udft

        if not 1 <= dim <= len(ishape):
            raise ValueError(f"`dim`={dim} must be in [1, len(ishape)={len(ishape)}]")
        if ir.ndim < dim:
            raise ValueError(f"`ir` must have at least `dim`={dim} dimensions")
        # Leading axes of `ir` are batch axes broadcast against the batch axes
        # of the input (e.g. one kernel per channel). They must not change the
        # output batch shape, otherwise the adjoint would have to sum over them.
        batch = tuple(ishape[: len(ishape) - dim])
        try:
            broadcast_ok = np.broadcast_shapes(ir.shape[:-dim], batch) == batch
        except ValueError:
            broadcast_ok = False
        if not broadcast_ok:
            raise ValueError(
                f"`ir` batch shape {ir.shape[:-dim]} must broadcast to the input "
                f"batch shape {batch}"
            )

        # oshape: batch dims are unchanged, convolved dims shrink by the kernel
        # size (K → N-K+1, valid convolution).
        #
        # Example: ishape=(B, H, W), ir.shape=(Kh, Kw) or (B, Kh, Kw), dim=2
        #
        # oshape = (B, H-Kh+1, W-Kw+1)
        kshape = ir.shape[-dim:]
        if any(k > n for (n, k) in zip(ishape[-dim:], kshape)):
            raise ValueError(
                f"`ir` shape {kshape} must not exceed the input shape "
                f"{tuple(ishape[-dim:])} on the convolved axes"
            )
        super().__init__(
            ishape=ishape,
            oshape=batch + tuple(n - k + 1 for (n, k) in zip(ishape[-dim:], kshape)),
            name=name,
        )

        self.dim = dim
        self.imp_resp = ir
        self.freq_resp = self._udft.ir2fr(ir, self.ishape[-dim:])

        self.margins = kshape
        # Extract the valid part of the circular convolution result.
        # The kernel is centered at K//2 (ifftshift convention), so the
        # circular output is yc[n] = Σₖ h[k]·x[n - k + K//2] while the valid
        # output is yv[m] = Σₖ h[k]·x[m + K-1 - k] = yc[m + K-1 - K//2]. For
        # each spatial axis of size N with kernel size K:
        #   start = K-1 - K//2 = (K-1)//2
        #   end   = start + (N - K + 1)
        # For even K, start = K//2 - 1 (not K//2, which would include a
        # wrapped-around sample). The batch axes use slice(None).
        self._slices: tuple = tuple(
            [slice(None)] * len(batch)
            + [
                slice((k - 1) // 2, (k - 1) // 2 + n - k + 1)
                for (n, k) in zip(ishape[-dim:], kshape)
            ]
        )

    def _dft(self, point: Array) -> Array:
        return self._udft.rdftn(point, self.dim)

    def _idft(self, point: Array) -> Array:
        return self._udft.irdftn(point, self.ishape[-self.dim :])

    def forward(self, point: Array) -> Array:
        return self._idft(self._dft(point) * self.freq_resp)[self._slices]

    def adjoint(self, point: Array) -> Array:
        xp = arr_api.get_namespace(self.freq_resp)
        out = xp.zeros(self.ishape, dtype=point.dtype, device=arr_api.device(point))
        if arr_api.is_jax_array(out):
            out = out.at[self._slices].set(point)
        else:
            out[self._slices] = point
        return self._idft(self._dft(out) * xp.conj(self.freq_resp))


class DirectConv(LinOp):
    """Direct convolution.

    The convolution is performed on the last N axes where N = ir.ndim.

    Parameters
    ----------
    ir : Array
        The impulse response (numpy array).
    ishape : tuple of int
        The shape of the input array.
    name : str, optional
        Name of the operator.

    Raises
    ------
    ValueError
        If `ir` has more axes than the input, or if it is larger than the input
        on a convolved axis.

    Notes
    -----
    Numpy-only. Uses `scipy.signal.oaconvolve` (Overlap-Add method), which is
    generally faster than FFT-based convolution when one array is much larger
    than the other. Requires scipy.
    """

    def __init__(self, ir: Array, ishape: Shape, name: str = "DConv"):

        if spsignal is None:
            raise ImportError("scipy is required for DirectConv")

        if ir.ndim > len(ishape):
            raise ValueError(
                f"`ir.ndim` ({ir.ndim}) must not exceed `len(ishape)` ({len(ishape)})"
            )
        # Otherwise `oaconvolve` in "valid" mode swaps `ir` and the input
        if any(k > n for (n, k) in zip(ishape[-ir.ndim :], ir.shape)):
            raise ValueError(
                f"`ir` shape {ir.shape} must not exceed the input shape "
                f"{tuple(ishape[-ir.ndim :])} on the convolved axes"
            )

        oshape = tuple(
            (
                ishape[idx]
                if idx < len(ishape) - len(ir.shape)
                else ishape[idx] - ir.shape[idx - (len(ishape) - len(ir.shape))] + 1
            )
            for idx in range(len(ishape))
        )

        super().__init__(
            ishape=ishape,
            oshape=oshape,
            name=name,
        )

        self._conv = spsignal.oaconvolve
        self._imp_resp = np.asarray(ir)
        # Leading singleton axes to broadcast against the batch axes of input
        self._ir = np.reshape(
            self._imp_resp, (len(self.ishape) - ir.ndim) * (1,) + ir.shape
        )

    @property
    def ir(self) -> Array:
        """The impulse response, as given at construction."""
        return self._imp_resp

    def forward(self, point: Array) -> Array:
        return self._conv(point, self._ir, mode="valid")

    def adjoint(self, point: Array) -> Array:
        # Correlation with `ir`: convolution with the conjugate flipped `ir`
        return self._conv(point, np.conj(np.flip(self._ir)), mode="full")


class FreqFilter(Diag):
    """Frequency filter in Fourier space.

    A `Diag` on the half-spectrum of a real array, as returned by
    `udft.rdftn`, defined by an impulse response.

    Parameters
    ----------
    ir : Array
        The real impulse response.
    shape : tuple of int
        The shape of the real array in direct space, used to compute the
        frequency response. It is not the `ishape` of the operator: see Notes.
    name : str, optional
        Name of the operator.

    Attributes
    ----------
    diag : Array
        The frequency response of the filter, from `udft.ir2fr(ir, shape)`.

    Raises
    ------
    ValueError
        If `ir` is complex (from `udft.ir2fr`). On the full spectrum of
        `udft.dftn`, use ``Diag(udft.ir2fr(ir, shape, onesided=False))``.

    Notes
    -----
    The operator applies to the half-spectrum, so its `ishape` and `oshape`
    are ``shape[:-1] + (shape[-1] // 2 + 1,)``, not ``shape``.
    For instance ``FreqFilter(ir, (8, 10)).ishape == (8, 6)``. Use it between
    `udft.rdftn` and `udft.irdftn`, as `CircConv` does. If you have the
    frequency response, just use Diag.

    """

    def __init__(self, ir: Array, shape: Shape, name: str = "Filter"):
        # `udft.ir2fr` concatenates `shape` with tuples: it may be a list
        super().__init__(udft.ir2fr(ir, tuple(shape)), name=name)


class CircConv(LinOp):
    """Circulant (periodic) convolution.

    Parameters
    ----------
    imp_resp : Array
        The real impulse response, with the same number of dimensions as
        `shape`. Its origin is at the center (see `udft.ir2fr`).
    shape : tuple of int
        Shape of the input and output arrays.
    name : str, optional
        Name of the operator.

    Attributes
    ----------
    imp_resp : Array
        The impulse response.

    Raises
    ------
    ValueError
        If ``imp_resp.ndim != len(shape)``. A 1D convolution of a 2D array,
        for instance, needs an `imp_resp` of shape ``(1, K)``. Or if
        `imp_resp` is larger than `shape` on an axis. If `imp_resp` is complex
        (from `udft.ir2fr`): use ``F.H @ Diag(udft.ir2fr(imp_resp, shape,
        onesided=False)) @ F`` with ``F = DFT(shape, len(shape))``.
    """

    def __init__(self, imp_resp: Array, shape: Shape, name: str = "CConv"):
        if imp_resp.ndim != len(shape):
            raise ValueError(
                f"`imp_resp.ndim` ({imp_resp.ndim}) must equal `len(shape)` "
                f"({len(shape)})"
            )
        if any(k > n for (n, k) in zip(shape, imp_resp.shape)):
            raise ValueError(
                f"`imp_resp` shape {imp_resp.shape} must not exceed `shape` "
                f"{tuple(shape)}"
            )

        self._udft = udft

        self.imp_resp = imp_resp
        self.ffilter = FreqFilter(imp_resp, shape)

        super().__init__(ishape=shape, oshape=shape, name=name)

    @property
    def freq_resp(self) -> Array:
        """The frequency response."""
        return self.ffilter.diag

    def _dft(self, arr: Array) -> Array:
        return self._udft.rdftn(arr, len(self.ishape))

    def _idft(self, arr: Array) -> Array:
        return self._udft.irdftn(arr, self.oshape)

    def forward(self, point: Array) -> Array:
        return self._idft(self.ffilter.forward(self._dft(point)))

    def adjoint(self, point: Array) -> Array:
        return self._idft(self.ffilter.adjoint(self._dft(point)))

    def fwadj(self, point: Array) -> Array:
        return self._idft(self.ffilter.fwadj(self._dft(point)))


class Diff(LinOp):
    """Difference operator.

    Compute the first-order differences along an axis.

    Parameters
    ----------
    axis : int
        The axis along which to perform the diff.
    ishape : tuple of int
        The shape of the input array.
    name : str, optional
        Name of the operator.

    Attributes
    ----------
    axis : int
        The axis along which the differences are performed.
    """

    def __init__(self, axis: int, ishape: Shape, name: str = "Diff"):
        oshape = list(ishape)
        oshape[axis] = ishape[axis] - 1
        super().__init__(ishape, tuple(oshape), name=name + f"[{axis}]")
        self.axis = axis

    def forward(self, point: Array) -> Array:
        """The forward application `A·x`.

        This corresponds to the application of the following matrix in 1D::

            -1  1  0  0
             0 -1  1  0
             0  0 -1  1

        """
        xp = arr_api.get_namespace(point)
        return xp.diff(point, axis=self.axis)

    def adjoint(self, point: Array) -> Array:
        """The adjoint application `Aᴴ·y`.

        This corresponds to the application of the following matrix in 1D::

            -1  0  0
             1 -1  0
             0  1 -1
             0  0  1
        """
        xp = arr_api.get_namespace(point)
        # `prepend` and `append` must be arrays (array API), with the dtype of
        # `point` (a Python 0 is cast to int64 by numpy, then float32 → float64)
        shape = list(point.shape)
        shape[self.axis] = 1
        zero = xp.zeros(tuple(shape), dtype=point.dtype, device=arr_api.device(point))
        return -xp.diff(point, prepend=zero, append=zero, axis=self.axis)


class Sampling(LinOp):
    """Sampling operator using numpy fancy indexing.

    Numpy-only. Index is a tuple of index arrays as in numpy fancy indexing.

    Parameters
    ----------
    ishape : tuple of int
        The shape of the input array.
    index : tuple of array_like of int
        Tuple of index arrays, one per dimension, as in numpy fancy indexing.
        Negative indices count from the end. The arrays are broadcast
        together and their broadcast shape is the output shape.

    Attributes
    ----------
    index : tuple of array of int
        The index arrays, with negative indices made positive.

    Raises
    ------
    ValueError
        If there is not one index array per dimension of `ishape`, or if an
        index is out of bounds.
    """

    def __init__(self, ishape: Shape, index: tuple):
        ishape = tuple(ishape)
        if len(index) != len(ishape):
            raise ValueError(
                f"`index` must have one array per dimension of `ishape`: "
                f"{len(index)} for {len(ishape)}"
            )
        # Positive indices since `np.ravel_multi_index` in `adjoint` refuses
        # negative ones, unlike indexing in `forward`
        self.index = tuple(
            np.where(idx < 0, idx + n, idx)
            for idx, n in zip((np.asarray(i) for i in index), ishape)
        )
        for axis, (idx, n) in enumerate(zip(self.index, ishape)):
            if np.any((idx < 0) | (idx >= n)):
                raise ValueError(f"`index` out of bounds on axis {axis} of size {n}")
        super().__init__(
            ishape,
            np.broadcast_shapes(*(idx.shape for idx in self.index)),
            name="Sampling",
        )

    def forward(self, point: Array) -> Array:
        return point[self.index]

    def adjoint(self, point: Array) -> Array:
        # Alternative that needs to be tested: out = np.zeros(self.ishape,
        # dtype=point.dtype); np.add.at(out, self.index, point)

        flat_index = np.ravel_multi_index(self.index, self.ishape)
        flat = flat_index.ravel()
        minlength = np.prod(self.ishape)
        w = point.ravel()
        if np.iscomplexobj(point):
            result = np.bincount(
                flat, weights=w.real, minlength=minlength
            ) + 1j * np.bincount(flat, weights=w.imag, minlength=minlength)
        else:
            result = np.bincount(flat, weights=w, minlength=minlength)
        # bincount always returns float64
        return np.reshape(result, self.ishape).astype(point.dtype, copy=False)


class Slice(LinOp):
    """Equivalent to obj[::2, 1, ...] etc.

    Parameters
    ----------
    ishape : tuple of int
        The shape of the input array.
    idx : index expression
        The index expression to apply (use ``np.index_exp`` to build it).
        The output shape is the shape of ``x[idx]`` for ``x`` of shape ``ishape``.
    See Also
    --------
    Sampling : when you have an array of indices that can handle multiple
        sampling of the same value.

    Notes
    -----
    Use `np.index_exp` to build the `idx` argument.

    Examples
    --------
    >>> s = Slice((10, 10), idx=np.index_exp[::2, 1])
    >>> y = s.forward(np.empty((10, 10)))  # shape (5,)
    >>> x = s.adjoint(y)                   # shape (10, 10)
    """

    def __init__(self, ishape: Shape, idx: tuple):
        # A broadcast view: no allocation of `ishape`, only of the output for
        # advanced indexing.
        oshape = np.broadcast_to(np.int8(0), ishape)[idx].shape
        super().__init__(ishape, oshape, name=f"S[{idx}]")
        self._idx = idx

    @property
    def idx(self) -> tuple:
        """The index expression."""
        return self._idx

    def forward(self, point: Array) -> Array:
        return point[self._idx]

    def adjoint(self, point: Array) -> Array:
        xp = arr_api.get_namespace(point)
        out = xp.zeros(self.ishape, dtype=point.dtype, device=arr_api.device(point))
        if arr_api.is_jax_array(out):
            out = out.at[self._idx].set(point)
        else:
            out[self._idx] = point
        return out


def _check_orthogonal(wavelet: str, pywt_module: types.ModuleType) -> None:
    """Raise `ValueError` if `wavelet` is not orthogonal.

    The inverse transform of pywt is the adjoint only for orthogonal wavelets.
    For biorthogonal ones (`biorX.Y`, `rbioX.Y`) the synthesis filters differ
    from the analysis ones, so the inverse is not the adjoint.
    """
    if not pywt_module.Wavelet(wavelet).orthogonal:
        raise ValueError(
            f"wavelet {wavelet!r} is not orthogonal: its inverse transform is "
            "not the adjoint. Use an orthogonal family (haar, dbN, symN, coifN)."
        )


class DWT(LinOp):
    """Unitary Discrete Wavelet Transform.

    Parameters
    ----------
    shape : tuple of int
        The input shape.
    level : int, optional
        The decomposition level. All sizes of `shape` must be divisible by
        `2**level`. Defaults to the largest level allowed by `shape` and the
        wavelet filter length.
    wavelet : str, optional
        The wavelet to use. Must be orthogonal.
    name : str, optional
        Name of the operator.

    Raises
    ------
    ValueError
        If `wavelet` is not orthogonal, or if a size of `shape` is not
        divisible by `2**level`.

    Attributes
    ----------
    wlt : str
        The wavelet.
    lvl : int
        The decomposition level.

    Notes
    -----
    NumPy-only. Uses pywt internally.
    """

    def __init__(
        self,
        shape: Shape,
        level: int | None = None,
        wavelet: str = "haar",
        name: str = "DWT",
    ):
        if pywt is None:
            raise ImportError("pywt is required for DWT")
        self._pywt = pywt
        _check_orthogonal(wavelet, self._pywt)

        # With periodization, each level halves the approximation. There are as
        # many coefficients as samples, and the transform is orthogonal, only if
        # every size stays even, that is is divisible by 2**level.
        dyadic = min((n & -n).bit_length() - 1 for n in shape)  # powers of 2
        if level is None:
            level = min(self._pywt.dwtn_max_level(shape, wavelet), dyadic)
            if level == 0:
                raise ValueError(
                    f"no decomposition level is possible for `shape` {shape} "
                    f"with wavelet {wavelet!r} (odd or too small size)"
                )
        elif any(n % 2**level for n in shape):
            raise ValueError(
                f"all sizes of `shape` {shape} must be divisible by "
                f"2**level = {2**level}"
            )

        super().__init__(shape, shape, name=name)

        self.wlt = wavelet
        self.lvl = level
        self._mode = "periodization"
        self._slices = self._pywt.coeffs_to_array(
            self._pywt.wavedecn(
                np.empty(shape), wavelet=wavelet, mode="periodization", level=level
            )
        )[1]

    def forward(self, point: Array) -> Array:
        return self._pywt.coeffs_to_array(
            self._pywt.wavedecn(
                point, wavelet=self.wlt, mode=self._mode, level=self.lvl
            )
        )[0]

    def adjoint(self, point: Array) -> Array:
        return self._pywt.waverecn(
            self._pywt.array_to_coeffs(point, self._slices),
            wavelet=self.wlt,
            mode=self._mode,
        )

    def fwadj(self, point: Array) -> Array:
        return point


class Analysis2(LinOp):
    """2D analysis operator with stationary wavelet decomposition.

    Parameters
    ----------
    shape : tuple of (int, int)
        The input shape.
    level : int
        The decomposition level.
    wavelet : str, optional
        The wavelet to use. Must be orthogonal.
    name : str, optional
        Name of the operator.

    Raises
    ------
    ValueError
        If `wavelet` is not orthogonal, or if `shape` is not 2D.

    Notes
    -----
    NumPy-only. Uses pywt internally. The output is a 3D array where the first
    axis is the coefficient axis, with the approximation coefficients at index 0
    and the detail coefficients at indices 1 to 3*level. The second and third
    axes are the spatial axes. See `pywt.swt2` documentation for more details on
    the output format.

    """

    def __init__(
        self,
        shape: tuple[int, int],
        level: int,
        wavelet: str = "haar",
        name: str = "A",
    ):
        if pywt is None:
            raise ImportError("pywt is required for Analysis2")
        self._pywt = pywt
        _check_orthogonal(wavelet, self._pywt)

        shape = tuple(shape)  # tuple concatenation below, `shape` may be a list
        if len(shape) != 2:
            raise ValueError(f"`shape` {shape} must be 2D")
        super().__init__(shape, (3 * level + 1,) + shape, name=name)
        self.wlt = wavelet
        self.lvl = level
        self.norm = True

    def forward(self, point: Array) -> Array:
        coeffs = self._pywt.swt2(
            point, wavelet=self.wlt, level=self.lvl, norm=self.norm, trim_approx=True
        )
        return self.coeffs2cube(coeffs)

    def adjoint(self, point: Array) -> Array:
        return self._pywt.iswt2(self.cube2coeffs(point), self.wlt, norm=self.norm)

    def cube2coeffs(self, point: Array) -> list:
        """Return pywt coefficients from 3D array."""
        # Indexing, not `np.squeeze`, that would also remove a spatial axis of
        # size 1
        coeffs_list: list = [point[0]]
        for lvl in range(self.lvl):
            coeffs_list.append(
                [point[3 * lvl + 1], point[3 * lvl + 2], point[3 * lvl + 3]]
            )
        return coeffs_list

    @staticmethod
    def coeffs2cube(coeffs: list) -> Array:
        """Return 3D array from pywt coefficients."""
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

    def im2coeffs(self, point: Array) -> list:
        """Return pywt coefficients from an image array."""
        split = np.split(point, 3 * self.lvl + 1, axis=1)
        coeffs_list: list = [split[0]]
        for lvl in range(self.lvl):
            coeffs_list.append(
                [split[3 * lvl + 1], split[3 * lvl + 2], split[3 * lvl + 3]]
            )
        return coeffs_list

    @staticmethod
    def coeffs2im(coeffs: list) -> Array:
        """Return an image array from pywt coefficients."""
        clist = [coeffs[0]]
        for coeff in coeffs[1:]:
            clist.extend([coeff[0], coeff[1], coeff[2]])
        return np.concatenate(clist, axis=1)

    def cube2im(self, cube: Array) -> Array:
        """Convert 3D coefficient cube to image-stacked array."""
        return self.coeffs2im(self.cube2coeffs(cube))

    def im2cube(self, im: Array) -> Array:
        """Convert image-stacked array to 3D coefficient cube."""
        return self.coeffs2cube(self.im2coeffs(im))

    def get_irs(self) -> Array:
        """Return the impulse response of the filter bank."""
        iarr = np.zeros(self.ishape)
        iarr[0, 0] = 1
        return self.forward(iarr)

    def get_frs(self) -> Array:
        """Return the frequency response of the filter bank."""
        return np.ascontiguousarray(
            np.fft.rfftn(self.get_irs(), self.ishape[-2:], axes=(-2, -1))
        )


class Synthesis2(LinOp):
    """2D synthesis operator with stationary wavelet decomposition.

    Parameters
    ----------
    shape : tuple of (int, int)
        The input shape.
    level : int
        The decomposition level.
    wavelet : str, optional
        The wavelet to use. Must be orthogonal.
    name : str, optional
        Name of the operator.

    Raises
    ------
    ValueError
        If `wavelet` is not orthogonal, or if `shape` is not 2D.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        level: int,
        wavelet: str = "haar",
        name: str = "S",
    ):
        self.analysis = Analysis2(shape, level, wavelet)
        super().__init__(self.analysis.oshape, self.analysis.ishape, name=name)
        self.wlt = self.analysis.wlt
        self.lvl = self.analysis.lvl

    def forward(self, point: Array) -> Array:
        return self.analysis.adjoint(point)

    def adjoint(self, point: Array) -> Array:
        return self.analysis.forward(point)

    def cube2coeffs(self, point: Array) -> list:
        """Return pywt coefficients from 3D array."""
        return self.analysis.cube2coeffs(point)

    def coeffs2cube(self, coeffs: list) -> Array:
        """Return 3D array from pywt coefficients."""
        return self.analysis.coeffs2cube(coeffs)

    def im2coeffs(self, point: Array) -> list:
        """Return pywt coefficients from image."""
        return self.analysis.im2coeffs(point)

    def coeffs2im(self, coeffs: list) -> Array:
        """Return image from pywt coefficients."""
        return self.analysis.coeffs2im(coeffs)

    def cube2im(self, cube: Array) -> Array:
        """Convert 3D coefficient cube to image-stacked array."""
        return self.analysis.cube2im(cube)

    def im2cube(self, im: Array) -> Array:
        """Convert image-stacked array to 3D coefficient cube."""
        return self.analysis.im2cube(im)

    def get_irs(self) -> Array:
        """Return the impulse response of the filter bank."""
        # The adjoint of the circular convolution by `h` is the circular
        # convolution by `h[-n mod N]`: a flip then a shift of one sample.
        return np.roll(np.flip(self.analysis.get_irs(), axis=(1, 2)), 1, axis=(1, 2))

    def get_frs(self) -> Array:
        """Return the frequency response of the filter bank."""
        return np.ascontiguousarray(
            np.fft.rfftn(self.get_irs(), self.ishape[-2:], axes=(-2, -1))
        )
