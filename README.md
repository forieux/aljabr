# Al-Jabr: Interfaces for implicit linear operators

This package implements interfaces for implicit linear operators, those defined
by function and callable instead of matrix. It is useful when the matrix
reprensentation is not adequat, for instance

- when the dimension is large,
- when more efficient computation is available instead of basic matrix vector
  product,
- when the vector interface is not easy to manipulate.

A typical example is the Discrete Fourier Transform that is a linear operator
but available through the usual `fft` and `ifft` function.

**The code is in early development stage, Pre-Alpha.**

If you are having issues, please let me know

francois.orieux AT universite-paris-saclay.fr

## Features

- A base type `LinOp` that represent an operator `A` whose behaviour depends on
  the two methods `forward` for `A·x`, that apply `A` on a vector `x`, and
  `adjoint` for `Aᴴ·x` that apply the adjoint `Aᴴ` on a vector `x`.
- Compatible with `LinearOperator` of scipy.
- The base type `LinOp` comes with handy utilities like `+`, `-` or `*`
  interface for basic composition, or automatic timing of instance creation or
  methods call.
- `matvec`, `rmatvec`, `__call__`, `*` and `@` interfaces.
- Instance of common linear operator like `Identity`, `Diagonal`, `DFT` or
  convolution.
- Utilities functions like `asmatrix` or `dottest`.

## Installation and documentation

The package is not actually on pypi but versionned, I recommend to use
[poetry](https://python-poetry.org/) and run in a terminal
```
poetry add  "git+https://github.com/forieux/aljabr.git"
```
or add
```
aljabr = {git = "https://github.com/forieux/aljabr.git", rev = "main"}
```
in the `[tool.poetry.dependencies]` section of your `pyproject.toml`.

`aljabr` depends on NumPy, SciPy, [udft](https://udft.readthedocs.io/) and
[PyWavelets](https://pywavelets.readthedocs.io/).

## License

The code is in the public domain.
