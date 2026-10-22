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

### Removed

- `LinOpLike` protocol: operator overloads (`+`, `-`, `*`, `@`) now only
  compose `LinOp` subclasses. Duck types were never fully supported.

### Added

- Add Iterator protocol to HStack and VStack.
- `llms.txt` and an `aljabr` agent skill (`skills/aljabr/SKILL.md`).
- `device` parameter to `dottest` and `fwadjtest`.
- `herm` function returning the conjugate transpose `Mᴴ`.
- `rtol` parameter to `is_*`.

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
