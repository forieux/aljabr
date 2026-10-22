---
name: aljabr
description: Write, compose and test implicit linear operators with the Python library aljabr (LinOp, forward/adjoint, DFT, Conv, Diff, Diag, VStack, dottest, …). Use when code imports `aljabr`, when the user asks to build a linear operator, a forward model, an adjoint or a Gram operator AᴴA defined by functions instead of a matrix, or wants to plug such operators into scipy.sparse.linalg solvers (cg, lsqr, eigsh) for inverse problems, deconvolution or image restoration.
---

# Using aljabr

aljabr gives an interface for **implicit linear operators**: a linear map `A`
defined by `forward` (`A·x`) and `adjoint` (`Aᴴ·y`) functions rather than a
matrix. Operators act on arrays of any shape and compose like matrices. The
library is deliberately minimal: no solver, no `A / y`, no automatic algorithm
choice.

Install: `pip install aljabr` (extras `aljabr[scipy]` for `DirectConv`/`fcond`,
`aljabr[wavelet]` for `DWT`/`Analysis2`/`Synthesis2`). Python ≥ 3.12. Every
public name is importable from the top level: `import aljabr` or
`from aljabr import LinOp, Diff, dottest`.

## Core model

- `ishape`, `oshape`: tuples, the **array** shapes of input and output, fixed at
  construction. A `DFT` on 256×256 is a different object from one on 512×512.
- `isize`, `osize`: number of elements. `shape == (osize, isize)` is the shape of
  the equivalent matrix (used by SciPy). `name` is a label used in `repr`.
- `forward(x)` takes an `ishape` array and returns an `oshape` array.
  `adjoint(y)` does the reverse. `fwadj(x)` returns `Aᴴ·A·x` (default
  `adjoint(forward(x))`).
- `matvec` / `rmatvec`: same as `forward` / `adjoint` but on `(N, 1)` column
  vectors (reshape in, flatten out).

## Applying and composing

```python
import numpy as np
import aljabr as aj

D = aj.Diff(axis=0, ishape=(64, 64))   # (64, 64) → (63, 64)
x = np.random.randn(64, 64)

y = D(x)            # == D.forward(x) == D * x
u = y * D           # == D.adjoint(y)  (array on the left → adjoint)
y_col = D @ x.reshape(-1, 1)            # matvec on a column vector
u_col = y_col.reshape(-1, 1) @ D        # rmatvec  — or D.rmatvec(y_col)

A = D.H             # Adjoint, a LinOp: A.forward is D.adjoint
D.H.H is D          # True, no wrapper chain
G = D.H @ D         # Symmetric (Gram), uses D.fwadj — same as D.G
P = D @ aj.Identity((64, 64))           # ProdOp (also D * Op)
S = 0.5 * D         # Scaled
R = D.G + 0.1 * aj.Diff(1, (64, 64)).G  # AddOp ; `-` gives SubOp
V = aj.VStack([aj.Diff(0, (64, 64)), aj.Diff(1, (64, 64))])  # x → [D0x; D1x]
```

Composition rules (a `ValueError` is raised otherwise):

- `A + B`, `A - B`: same `ishape` **and** same `oshape` (equal `shape` is not
  enough).
- `A @ B` / `A * B`: `A.ishape == B.oshape`.
- `VStack(ops)`: all share `ishape`; output is a `(Σ osize, 1)` column vector.
  `V.apply(x)` gives the list of per-operator outputs, `V.split(y)` cuts a
  column vector back into shaped pieces, `len(V)`, `V[i]` and iteration work.
  `V.H` is an `HStack` of the adjoints.
- `HStack(ops)`: all share `oshape`; input is a `(Σ isize, 1)` column vector
  (use `aj.vectorize([x0, x1])` / `aj.unvectorize(x, shapes)`).

`A.H @ A` and `A @ A.H` are detected and return a `Symmetric` instead of a
generic `ProdOp`. Only `A.H @ A` (or `A.G`) benefits from an overridden
`A.fwadj`. For a `Symmetric`, `.H` returns itself.

## Writing a new operator

Subclass `LinOp`, call `super().__init__` with the shapes, implement `forward`
and `adjoint`. Override `fwadj` only if `AᴴA` has a cheaper formula.

```python
import numpy as np
from aljabr import LinOp, Array, Shape


class Mask(LinOp):
    """Keep pixels where `mask` is True, zero elsewhere."""

    def __init__(self, mask: Array, name: str = "M"):
        super().__init__(ishape=mask.shape, oshape=mask.shape, name=name)
        self.mask = mask

    def forward(self, x: Array) -> Array:
        return np.where(self.mask, x, 0)

    def adjoint(self, y: Array) -> Array:
        return np.where(self.mask, y, 0)

    def fwadj(self, x: Array) -> Array:
        return self.forward(x)  # MᴴM = M for a mask
```

For a one-off operator, `BaseOp` avoids the class:

```python
F = aj.BaseOp(
    forward=lambda x: np.fft.fft2(x, norm="ortho"),
    adjoint=lambda y: np.fft.ifft2(y, norm="ortho"),
    ishape=(64, 64), oshape=(64, 64),
    fwadj=lambda x: x,  # optional
    name="F",
)
```

Other building blocks: `Symmetric(forward, shape)` for a self-adjoint operator
given by one function, `Dense(matrix, ishape=None, oshape=None)` to wrap an
explicit 2D matrix.

Subclasses are automatically instrumented: `forward`, `adjoint`, `fwadj` emit a
`UserWarning` on wrong input/output shapes, and timings go to
`A.metadata["init"]` (float) and `A.metadata["forward" | "adjoint" | "fwadj"]`
(list of durations). Do not reimplement these.

**Always validate** a new operator:

```python
assert aj.dottest(M, num=5)     # ⟨Aᴴu, v⟩ == ⟨u, Av⟩ — checks adjoint
assert aj.fwadjtest(M, num=5)   # fwadj(x) == adjoint(forward(x))
```

Both accept `rtol`, `atol`, `echo=True` (print both sides), `xp=` (array
namespace for the random test vectors, e.g. `xp=torch`) and `device=` (their
device, e.g. `device="cuda"` for an operator on GPU).

## Concrete operators

| Operator | Signature | Shapes / notes |
|---|---|---|
| `Identity` | `(shape)` | `shape → shape` |
| `Diag` | `(diag)` | element-wise product, `diag.shape → diag.shape` |
| `DFT` | `(shape, ndim)` | unitary DFT on last `ndim` axes, complex output, `fwadj` = identity |
| `RealDFT` | `(shape, ndim)` | `DFT` of a real input on the half spectrum, last axis → `n // 2 + 1`; adjoint = inverse, **not** the standard adjoint (see pitfalls) |
| `Conv` | `(ir, ishape, dim)` | FFT convolution on last `dim` axes, **"valid"** output (`n - k + 1`); adjoint is "full" |
| `DirectConv` | `(ir, ishape)` | same as `Conv` with `scipy.signal.oaconvolve`; NumPy only, needs scipy |
| `CircConv` | `(imp_resp, shape)` | periodic convolution, `shape → shape` |
| `FreqFilter` | `(ir, ishape)` | `Diag` in Fourier space from an impulse response (use `Diag` if you already have the frequency response) |
| `Diff` | `(axis, ishape)` | first-order difference, `oshape[axis] = ishape[axis] - 1` |
| `Slice` | `(ishape, idx)` | `x[idx]`, build `idx` with `np.index_exp[::2, 1]`; adjoint zero-fills |
| `Sampling` | `(ishape, index)` | NumPy fancy indexing, `index` = tuple of int arrays; adjoint accumulates repeated indices; NumPy only |
| `DWT` | `(shape, level=None, wavelet="haar")` | unitary nD wavelet transform (periodization); orthogonal wavelets only (haar, dbN, symN, coifN); every size divisible by `2**level` (default: largest valid level); needs pywt |
| `Analysis2` | `(shape, level, wavelet="haar")` | 2D stationary wavelet analysis, `(H, W) → (3·level+1, H, W)`; orthogonal wavelets only; needs pywt |
| `Synthesis2` | `(shape, level, wavelet="haar")` | adjoint of `Analysis2`, `(3·level+1, H, W) → (H, W)` |

All accept an optional `name=` (except `Sampling`, `Slice`). Check shapes with
`print(op)`, e.g. `Conv (Conv): (64, 64) → (60, 60)`.

## Utilities

- `dottest`, `fwadjtest`: see above.
- `allclose(a, b, rtol, atol)`: array-API version of `np.allclose`.
- `asmatrix(A, like=None)` / `A.asmatrix(like=None)`: dense `(osize, isize)`
  matrix by applying `forward` to every basis vector — **small operators
  only** (debug, tests). The matrix has the dtype returned by the operator
  (complex for `DFT`). Pass `like=` an array of the target backend, dtype and
  device for the basis vectors (default NumPy float64).
- `herm(M)`: conjugate transpose `Mᴴ` of an array.
- `is_sym`, `is_pos_def`, `is_semi_pos_def`, `is_neg_def`, `is_semi_neg_def`,
  `cond`: build the dense matrix — small operators only. `is_sym` tests
  `M = Mᴴ` (Hermitian). The `is_*_def` functions test the eigenvalues of
  `M + Mᴴ` (so `Re(xᴴMx)`), with `rtol=` relative to the largest eigenvalue
  (default `max(M.shape)·eps`). `cond` is `max|λ| / min|λ|`, the true
  condition number only for normal matrices.
- `fcond(A, tol=0.1)`: condition number estimated with Lanczos
  (`scipy.sparse.linalg.eigsh`), for large Hermitian **definite** operators
  (e.g. full-rank `A.G`); needs scipy.

## Solving with SciPy

A `LinOp` exposes `shape`, `matvec`, `rmatvec`, so it can be passed directly
where SciPy expects a `LinearOperator`. Work on flattened vectors and reshape
the result yourself.

```python
import scipy.sparse.linalg as spla

H = aj.Conv(np.ones((5, 5)) / 25, ishape=(64, 64), dim=2)   # blur, (64,64)→(60,60)
D0, D1 = aj.Diff(0, H.ishape), aj.Diff(1, H.ishape)
data = H(x_true) + 0.01 * np.random.randn(*H.oshape)

# Tikhonov: (HᴴH + μ(D0ᴴD0 + D1ᴴD1)) x = Hᴴ y
Q = H.G + 0.1 * (D0.G + D1.G)
sol, info = spla.cg(Q, H.adjoint(data).ravel())
x_hat = sol.reshape(H.ishape)
```

## Array API backends

`LinOp`, the algebra, `Identity`, `Diag`, `DFT`, `RealDFT`, `Conv`,
`CircConv`, `FreqFilter`, `Diff`, `Slice` and `Dense` use the array namespace
of their inputs (or of the array given at construction: `diag`, `ir`,
`matrix`), so they work with NumPy, PyTorch, JAX, CuPy… JAX in-place updates
are handled. `DirectConv`, `Sampling`, `DWT`, `Analysis2`, `Synthesis2` and
`fcond` are NumPy only. `Conv`, `CircConv` and `FreqFilter` rely on
`udft.ir2fr`, which creates the frequency response on the default device: an
impulse response on another device (e.g. GPU) currently fails. In your own `forward`/`adjoint`, either use one library
or `array_api_compat.get_namespace(x)` to stay agnostic.

## Pitfalls

- Scalars for `s * A` must be Python `int`, `float`, `complex` (or
  `np.float64`, a `float` subclass). A `np.float32`, 0-d array or tensor
  scalar is treated as an array and calls `A.adjoint(s)`: write
  `float(s) * A`.
- `x * A` (array on the left) is the **adjoint**, not the forward. `A @ x`
  expects a column vector `(N, 1)`, not an `ishape` array: use `A(x)` on
  shaped arrays.
- Shape mismatches only warn; treat any `UserWarning` from aljabr as a bug.
- `Conv` and `DirectConv` reduce the size ("valid"); use `CircConv` for a
  same-size periodic model. `Diff` also loses one sample on its axis.
- `A @ A.H` returns a `Symmetric` built from `A.H.fwadj`, which is the
  generic `A·Aᴴ` (no optimized path).
- `RealDFT.adjoint` is the inverse (`irdftn`) and `fwadj` the identity, as
  for `DFT` on the full Hermitian spectrum. On the half spectrum this is the
  adjoint for a weighted inner product (weight 2 except frequencies 0 and
  `n/2`), not the standard one: `dottest(RealDFT(...))` fails by design. Use
  it as a change of representation (`Aᴴ·Λ·A`), not in generic solvers; use
  `DFT` there.
- Do not call `asmatrix`, `is_*`, or `cond` on large operators: they cost
  `isize` forward calls and an `osize × isize` dense matrix.

## References

- Guide: https://raw.githubusercontent.com/forieux/aljabr/main/docs/guide.md
- Docs: https://aljabr.readthedocs.io/
- Sources: `src/aljabr/linop.py`, `src/aljabr/concrete.py`, `src/aljabr/utils.py`
