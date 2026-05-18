# Working with LinOp

This page walks through the main features of `LinOp` using a simple 2D unitary
FFT as a running example (see `concrete.DFT`). Every concept introduced here
applies to any operator you build on top of `LinOp` and are present or are used
in operators in `concrete`.

The idea is to be helpful, easy to use, and eliminate repetitive tasks — without
doing the work for you or thinking for you.

## Subclassing LinOp: `forward` and `adjoint`

The only requirement is to subclass `LinOp` and implement `forward` and
`adjoint`. The constructor must call `super().__init__` with `ishape` and
`oshape` — the shapes of the input and output arrays.

```python
import numpy as np
from aljabr import LinOp, Array, Shape

class FFT2(LinOp):
    """Unitary 2D FFT — maps an image to its Fourier coefficients."""

    def __init__(self, shape: Shape):
        super().__init__(ishape=shape, oshape=shape, name="F")

    def forward(self, x: Array) -> Array:
        return np.fft.fft2(x, norm="ortho")   # F·x

    def adjoint(self, y: Array) -> Array:
        return np.fft.ifft2(y, norm="ortho")  # Fᴴ·y
```

The unitary normalization (`norm="ortho"`) ensures $F^H F = I$: the FFT is an
isometry, so the adjoint equals the inverse.

### Optimising with `fwadj`

`fwadj` computes $F^H \cdot F \cdot x$ in one shot. By default it calls
`adjoint(forward(x))`, which is correct but not always optimal. When you know a
cheaper formula — as here, where $F^H F = I$ — override it:

```python
    def fwadj(self, x: Array) -> Array:
        return x   # Fᴴ·(F·x) = x since the FFT is unitary
```

This short-circuits two FFT calls into a no-op. The optimisation is optional but
matters in iterative algorithms that call `fwadj` thousands of times.

## Shapes: `ishape` and `oshape`

Shapes are fixed at construction and describe the **array** shape of the input
and output — not flattened vectors. They are required: this is a design decision
and different than `np.fft.fft2` function.

```python
F = FFT2((256, 256))  # Create the operator (like a matrix)
print(F.ishape)   # (256, 256)
print(F.oshape)   # (256, 256)

x = np.empty(F.ishape)
coef = F.forward(x)
np.allclose(F.adjoint(coef), x)
```

For a non-square operator (e.g., a difference along axis 0):

```python
from aljabr import Diff
D = Diff(axis=0, ishape=(256, 256))
print(D.ishape)   # (256, 256)
print(D.oshape)   # (255, 256)  — one fewer row after diff on axis 0.
```

`Diff` is a concrete operator that asks only for `ishape` at creation since
`oshape` can be deduced (as `FFT2` above).

## Attributes and derived properties

Once constructed, several attributes and read-only properties are available:

| Name     | Type              | Description                                |
|----------|-------------------|--------------------------------------------|
| `ishape` | `tuple[int, ...]` | Input array shape                          |
| `oshape` | `tuple[int, ...]` | Output array shape                         |
| `isize`  | `int`             | `prod(ishape)` — number of input elements  |
| `osize`  | `int`             | `prod(oshape)` — number of output elements |
| `shape`  | `tuple[int, int]` | `(osize, isize)` — matrix shape            |
| `name`   | `str`             | Human-readable label                       |

```python
print(F.isize)    # 65536  (= 256*256)
print(F.shape)    # (65536, 65536)
print(repr(F))    # F (FFT2): (256, 256) → (256, 256)
```

## Practical conveniences

### Calling the operator

`F(x)` is the same as `F.forward(x)`:

```python
y = F(x)          # equivalent to F.forward(x)
```

### Column-vector interface: `matvec` and `rmatvec`

`matvec` and `rmatvec` suppose column vectors for inputs and outputs, of shape
`(isize, 1)` and `(osize, 1)` respectively. They are useful when an algorithm
expects 1D vectors regardless of the operator's native shapes:

```python
x_col = x.reshape(-1, 1)          # (65536, 1)
y_col = F.matvec(x_col)           # y = Fx — forward on flat vector x
u_col = F.rmatvec(y_col)          # u = yᵀ F = Fᴴ y — adjoint on flat vector y
```

Internally, `matvec` reshapes to `ishape`, calls `forward`, then flattens the
result; `rmatvec` does the same through `adjoint`.

### String representation

`__repr__` returns a compact description

```python
repr(F)   # "F (FFT2): (256, 256) → (256, 256)"
```
with `name`, `class`, `ishape -> oshape`.

## SciPy compatibility

Any `LinOp` can be passed directly to SciPy functions that accept a
`LinearOperator` (e.g., `scipy.sparse.linalg.eigsh`, `cg`, `lsqr`). SciPy's
`aslinearoperator` recognises objects that expose `matvec`, `rmatvec`, and
`shape` — which every `LinOp` does:

```python
import scipy.sparse.linalg as spla

AtA = spla.aslinearoperator(F)          # wraps F transparently
vals = spla.eigsh(AtA, k=5, return_eigenvectors=False)
```

`LinOp` objects can also be passed directly without the wrapping step to
functions that call `aslinearoperator` internally.

## Algebraic operators

`LinOp` overloads the standard Python operators for transparent composition.

### `+` and `-`

```python
import numpy as np
from aljabr import Identity, Diag

I = Identity((256, 256))
W = Diag(np.full((256, 256), 0.5))   # element-wise ½

C = I + W   # AddOp — forward: x + 0.5·x
D = I - W   # SubOp — forward: x − 0.5·x
```

Both operands must share the same `ishape` and `oshape` (even if they have the same `shape`).

### `*` — apply or scale

When the right operand is an **array**, `*` applies the operator:

```python
y = F * x         # F.forward(x), same as F(x)
```

When the right operand is a **scalar**, `*` returns a scaled operator:

```python
G = 2.0 * F       # Scaled — G.forward(x) = 2 * F.forward(x)
```

When the right operand is a **LinOp**, `*` returns a composition:

```python
H = F * D         # ProdOp — H.forward(x) = F.forward(D.forward(x))
```

### `@` — matrix-multiply style

`@` behaves like `*` but uses `matvec`/`rmatvec` for array operands (column
vectors), and detects the special case $A^H \cdot A$:

```python
# On arrays (column vectors): uses matvec
y_col = F @ x_col      # F.matvec(x_col)

# On LinOp: composition
H = F @ D              # ProdOp

# Special case: FᴴF → Symmetric automatically
S = F.H @ F            # Symmetric (not a generic ProdOp)
S = F @ F.H            # also Symmetric
```

This automatic detection avoids wrapping $A^H A$ in a generic `ProdOp` when a
`Symmetric` is the right result.

## `asmatrix` — materialise the operator

`asmatrix()` builds the explicit dense matrix by applying `forward` to each
canonical basis vector. Useful for debugging and small operators:

```python
M = F.asmatrix()    # shape (65536, 65536) — very large for 256²!

# More realistic example
D_small = Diff(axis=0, ishape=(4, 4))
print(D_small.asmatrix())
```

The module-level function `asmatrix(linop)` works the same way and also accepts
plain arrays.

:::{warning}

`asmatrix` allocates an `(osize, isize)` matrix and calls `forward` `isize`
times. For a 256² image that is 65 536 calls and a 32 GB matrix. Use only for
small operators or debugging. Moreover, `asmatrix` assumes NumPy float64 by
default; if this is not the case, pass the `like` parameter.

:::

## Automatic timing and shape checking

Every `LinOp` subclass automatically gets two runtime behaviours applied to its
methods.

- **`timeit`** — wraps `__init__`, `forward`, `adjoint`, and `fwadj`. After
  each call the elapsed time is stored in the `metadata` dict on the instance.
  `forward` and `adjoint` also accumulate a full history in lists:

  ```python
  F = FFT2((256, 256))
  y = F.forward(x)
  print(F.metadata["forward_last_duration"])   # seconds, last call
  print(F.metadata["init_last_duration"])      # construction time
  print(F.metadata["forward_durations"])       # [t1, t2, …] — all calls
  print(F.metadata["adjoint_durations"])       # same for adjoint
  ```

- **`checkshape`** — wraps `forward`, `adjoint`, and `fwadj`. Emits a
  `UserWarning` if an input or output array has the wrong shape. This catches
  shape bugs early without raising hard errors.

Both decorators are applied transparently — you do not need to do anything in
your subclass.

## `BaseOp` — defining an operator without subclassing

Instead of subclassing you can use `BaseOp` with a pair of callables

```python
from aljabr import BaseOp

fft2_op = BaseOp(
    forward=lambda x: np.fft.fft2(x, norm="ortho"),
    adjoint=lambda y: np.fft.ifft2(y, norm="ortho"),
    ishape=(256, 256),
    oshape=(256, 256),
    name="F",
)
```

`BaseOp` accepts an optional `fwadj` callable for the same optimisation as
above. It is otherwise identical to a hand-written subclass.

## The adjoint: `Adjoint` and `.H`

`.H` returns the adjoint as a proper `LinOp`, not just a callable:

```python
Fh = F.H                   # Adjoint of F
y  = Fh.forward(x)         # = F.adjoint(x)
```

`Adjoint` is a singleton — it avoids creating a chain of wrappers:

```python
F.H.H is F                 # True — double adjoint cancels
```

For operators that know their own adjoint (`Symmetric`, `Adjoint`), `.H`
delegates to them directly rather than wrapping in a new `Adjoint`.

## `Symmetric` and `.S` — the normal operator $A^H A$

`.S` returns the operator $A^H A$ as a `Symmetric`:

```python
AtA = F.S                  # Symmetric wrapping F.fwadj
z   = AtA.forward(x)       # = F.fwadj(x)  (optimised path if overridden)
```

`Symmetric` enforces $A^H A = A$ at the type level: its `.H` property returns
`self`, and `Adjoint(A)` is `A` for any `Symmetric` instance.

### `@` and the automatic detection

The `@` operator detects the pattern $A^H \cdot A$ at composition time:

```python
AtA = F.H @ F    # Symmetric, not a ProdOp(Adjoint(F), F)
AtA = F @ F.H    # also Symmetric
```

This matters because `Symmetric.fwadj` is `forward` (no extra adjoint call),
and iterative solvers that call `fwadj` in a loop benefit from the saved work.
When you override `fwadj` in your subclass — as in the `FFT2` example above —
the `Symmetric` produced by `.S` or `@` automatically uses that override.

## Array-agnostic operators

`LinOp` is not tied to NumPy and uses the array API standard with the help of
`array-api-compat`. This is also the case for many derived `LinOp` and concrete
`Operator`.

As the author of `forward` and `adjoint`, you can use the array library you want
or even make them array-agnostic too. However, `asmatrix`, which builds canonical
basis vectors, needs to know the backend and dtype if it's different from Numpy
float64.
