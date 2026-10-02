"""Prewhitening of autocorrelated residuals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ar1_prewhiten(x: ArrayLike, rho: float) -> np.ndarray:
    """Innovations w_t = x_t - rho * x_{t-1} along axis 0; the first sample is dropped.

    A persistent mean shift mu in x appears as (1 - rho) * mu in w, and
    Var(w) = (1 - rho^2) * Var(x) for a stationary AR(1) x.
    """
    x = np.asarray(x, dtype=float)
    return x[1:] - rho * x[:-1]
