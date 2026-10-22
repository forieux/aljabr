# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- type checker annotations fix
- `LinOp.asmatrix` keeps the dtype returned by the operator.
- `Identity.asmatrix` honours the `like` dtype and device.
- `Diag.asmatrix`, `Dense.asmatrix` and `asmatrix(array)` honour the `like`
  device.
- `Diag.asmatrix` no longer uses the non-standard `xp.diag`.
- `Conv.adjoint` and `Slice.adjoint` return arrays on the input device.
- `is_*` correctly handle hermitian matrix.
- `is_pos_def` docstring: broken LaTeX (`\f` escape).
- `Conv` with an even kernel size
- require `udft>=4.1.0` for correct normalisation
- `Conv` with batch axes in `ir` (e.g. one kernel per channel)
- `RealDFT` docstring: its adjoint is the inverse
- `DWT`, `Analysis2` and `Synthesis2` raise `ValueError` for non-orthogonal
  wavelets.
- `DWT` raises `ValueError` if `shape` is not divisible by `2**level` (wrong
  number of coefficients). The default level is the largest compatible one.
- `DirectConv.adjoint` conjugates the impulse response (complex `ir`).
- `DirectConv.ir` returns the impulse response as given.
- `A * s`, `s * A`, `A @ s` and `s @ A` return a `Scaled`.
- `V.H.H is V` for VStack and HStack.
- `forward`, `adjoint` and `fwadj` of subclasses accept extra arguments.
- `Sampling.adjoint` keeps the input dtype.
- `FreqFilter` docstring: the operator acts on the half-spectrum.
- `Slice` no longer allocates an `ishape` array to infer `oshape`.
- Timings use `time.perf_counter`.
- `CircConv` raises `ValueError` if `imp_resp.ndim != len(shape)`.
- `Diff.adjoint` keeps the input dtype and works with array API namespaces
  other than numpy.
- `Synthesis2.get_irs` and `get_frs` were shifted by one sample.
- `Analysis2.get_frs` and `Synthesis2.get_frs` pass `axes` to `rfftn`.
- `VStack` and `HStack` accept any iterable of operators (e.g. a generator).
- `RealDFT`, `FreqFilter`, `CircConv`, `Analysis2` and `Synthesis2` accept a
  shape given as a list.

### Changed

- `FreqFilter` parameter `ishape` renamed `shape`: it is the direct-space
  shape, not the operator `ishape`.
- `metadata["forward" | "adjoint" | "fwadj"]` is a `deque` of the last
  `LinOp.max_timings` (1000) durations instead of an unbounded list.

### Removed

- `LinOpLike` protocol: operator overloads (`+`, `-`, `*`, `@`) now only
  compose `LinOp` subclasses. Duck types were never fully supported.

### Added

- Add Iterator protocol to HStack and VStack.
- `llms.txt` and an `aljabr` agent skill (`skills/aljabr/SKILL.md`).
- `device` parameter to `dottest` and `fwadjtest`.
- `herm` function returning the conjugate transpose `Mᴴ`.
- `rtol` parameter to `is_*`.
- `is_scalar` function: the scalar rule of `*` and `@`.

## [0.5.2] - 2026-05-19

### Fixed

- Typo and precision
- url in pyproject

## [0.5.1] - 2026-05-19

### Changed

- Published on pypi.

### Fixed

- Small fix in bump2version config

## [0.5.0] - 2026-05-19

### Changed

Major update of the library before switch to beta. Switch to __ini_subclass__, many bug fixing, many cleaning, many consolidation, add test, add docs.

## [0.4.0] - 2024-07-04

### Changed

Switch to « src » layout.

## [0.3.2] - 2022-09-01

### Changed

- More explicite message
- Fix `Slice` adjoint

### Added

- Add `Slice` operator

## [0.3.0] - 2022-06-20

### Changed

- Remove fftw dependency
- Small fix and typos
- Maximum python version 3.11

## [0.2.0] - 2022-03-15

### Changed

- Add py.typed to inform that the module is typed.
- Fix shape typing

## [0.1.0] - 2021-04-16

First public release.
