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
- Algebraic composition: `A + B`, `A - B`, `A @ B`, `scalar * A`, `A.H`, `A.G`
- Array-API compatible — works with NumPy, PyTorch, JAX, and others via
  [array-api-compat](https://data-apis.org/array-api-compat/)
- Compatible with `scipy.sparse.linalg.LinearOperator` — can also be passed
  directly as argument (wrap it with `aslinearoperator` for `eigs` and `eigsh`)
- Concrete operators: `Identity`, `Diag`, `DFT`, `RealDFT`, `Conv`,
  `DirectConv`, `CircConv`, `FreqFilter`, `Diff`, `Sampling`, `Slice`,
  wavelet, …
- Utilities: `dottest`, `fwadjtest`, `is_sym`, `is_pos_def`, `cond`, …


## Installation

```bash
# with poetry (or uv)
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
y = F.forward(x)  # F·x
# or
y = F * x
# or
y = F(x)

z = F.adjoint(y)  # Fᴴ·y
# or
z = y * F  # (= Fʰ·yᵗ)

# Algebraic composition
A = aljabr.Diff(axis=0, ishape=(256, 256))
B = A.H @ A  # AᴴA as a Symmetric operator

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

## Use with AI agents

- [`llms.txt`](https://aljabr.readthedocs.io/en/stable/llms.txt) follows the
  [llms.txt](https://llmstxt.org/) proposal and gives agents a map of the
  documentation.
- [`skills/aljabr`](https://github.com/forieux/aljabr/blob/main/skills/aljabr/SKILL.md) is an [Agent Skill](https://agentskills.io) that teaches agents the conventions of aljabr. In Claude Code, install it from this repository, which is also a plugin marketplace:
  ```
  /plugin install aljabr --marketplace forieux/aljabr
  ```
  or with `/plugin marketplace add forieux/aljabr` then `/plugin install
  aljabr@aljabr`. For other agents, copy the `skills/aljabr` directory into their
  skills directory.

## How AI is used

This project is not vibecoded. But with performant AI agents, I use them for bug
tracking and design reflection, not for code writing. I write the code, but the
one proposed by the agent is in the window on the side, obviously.

The agent and I write documentation. I reread the agent part, and the agent
reread my part, and I chose to commit. The agent writes all tests. I chose that
approach because I don't have the time to write all the documentation and tests.

I think this is a good trade-off for a project maintained by one person. It
allows me to propose a more polished version of what I publish, as if I were a
full-time developer, which is also good for my own use.

## License

MIT — see [LICENSE](https://github.com/forieux/aljabr/blob/main/LICENSE).
