"""Page's CUSUM (Page 1954), vectorized over replicas along axis 0."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def cusum(increments: ArrayLike) -> np.ndarray:
    """W_t = max(0, W_{t-1} + l_t), W_0 = 0, along axis 0.

    Computed as S_t - min(0, min_{k <= t} S_k), with S the partial sums of l.
    """
    s = np.cumsum(np.asarray(increments, dtype=float), axis=0)
    return s - np.minimum(np.minimum.accumulate(s, axis=0), 0.0)


def gaussian_llr(x: ArrayLike, shift: ArrayLike, sigma: ArrayLike = 1.0) -> np.ndarray:
    """Log-likelihood ratio of N(shift, sigma^2) against N(0, sigma^2).

    shift and sigma may vary in time (e.g. a load-dependent fault signature)
    and broadcast against x.
    """
    shift = np.asarray(shift, dtype=float)
    return shift / np.asarray(sigma, dtype=float) ** 2 * (np.asarray(x, dtype=float) - shift / 2.0)
