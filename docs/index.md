# aljabr — Implicit linear operators

**aljabr** is a small Python library providing an interface for implicit linear
operators: linear maps defined by `forward` and `adjoint` callables rather than
an explicit matrix. It is useful when an explicit matrix representation is not
adequate — for instance when the dimension is large, when a more efficient
algorithm exists, or when the operator is naturally expressed as a function.

A typical example is the Discrete Fourier Transform, which is a linear operator
available through `fft` and `ifft` functions. aljabr lets you treat it — and
compose it with others — as if it were a matrix.

The foundation is the `LinOp` type. Everything else — concrete operators,
utilities, algebraic composition — is built on top of it and optional. I made
this library for my own usage.

## Key differences from similar tools

**vs. `scipy.sparse.linalg.LinearOperator`** — input and output arrays can have
any shape, not just vectors. A 2D FFT operator maps an image to an image. Shapes
are fixed at construction: a `DFT` on a 512² image is a different object than
one on a 256² image. This is intentional.

**vs. PyLops** — aljabr is deliberately minimal. It does not provide solvers or
choose algorithms for you. There is no `x = A / y`. It stays out of your way,
offering only readability and convenience.

## Features

- Abstract base class `LinOp` with `forward` (`A·x`) and `adjoint` (`Aᴴ·y`)
- Algebraic composition: `A + B`, `A - B`, `A @ B`, `scalar * A`, `A.H`, `A.S`
- Array-API compatible — works with NumPy, PyTorch, JAX, and others via
  [array-api-compat](https://data-apis.org/array-api-compat/)
- Compatible with `scipy.sparse.linalg.LinearOperator` — can also be passed
  directly as argument
- Concrete operators: `Identity`, `Diag`, `DFT`, `RealDFT`, `Conv`,
  `DirectConv`, `CircConv`, `FreqFilter`, `Diff`, `Sampling`, `Slice`,
  wavelet, …
- Utilities: `dottest`, `fwadjtest`, `is_sym`, `is_pos_def`, `cond`, …


## Installation

```bash
# with uv (or uv)
poetry add aljabr

# with pip
pip install aljabr
```

Optional dependencies:

```bash
poetry add aljabr[wavelet]   # PyWavelets — required for DWT, Analysis2, Synthesis2
poetry add aljabr[scipy]     # SciPy — required for DirectConv and fcond
```

## Quick example

```python
import numpy as np
import aljabr

# Wrap the FFT as a linear operator
F = aljabr.DFT(shape=(256, 256), ndim=2)

x = np.random.randn(256, 256)
y = F.forward(x)        # F·x
z = F.adjoint(y)        # Fᴴ·y

# Algebraic composition
A = aljabr.Diff(axis=0, ishape=(256, 256))
B = A.H @ A             # AᴴA as a Symmetric operator

# Validate with the dot test
from aljabr import dottest
assert dottest(A, num=5)
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide
```

```{toctree}
:maxdepth: 2
:caption: API Reference

linop
concrete
utils
```

## Status

Pre-Alpha. The API may change. Feedback welcome:
<francois.orieux@universite-paris-saclay.fr>

## License

MIT — see [LICENSE](https://github.com/forieux/aljabr/blob/main/LICENSE).
