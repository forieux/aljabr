import numpy as np
from aljabr.linop import Dense

RNG = np.random.default_rng(42)


def make_dense(m: int, n: int) -> Dense:
    """Return an m×n Dense operator with random entries."""
    return Dense(RNG.standard_normal((m, n)), ishape=(n,), oshape=(m,))
