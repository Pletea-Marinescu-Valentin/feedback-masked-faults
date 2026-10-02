"""Effective (long-run) variance of autocorrelated residuals.

For a stationary Gaussian residual with autocovariances gamma_k, the
Kullback-Leibler information that n samples carry about a persistent mean
shift mu grows as n * mu^2 / (2 * sigma_eff^2), where
sigma_eff^2 = sum_k gamma_k = 2 * pi * f(0) is the long-run variance. For an
AR(1) residual with marginal variance sigma^2 and coefficient rho,
sigma_eff^2 = sigma^2 * (1 + rho) / (1 - rho): the information per sample, and
hence the CUSUM delay, changes by the factor (1 - rho) / (1 + rho) relative to
i.i.d. residuals of the same marginal variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ar1_long_run_variance(sigma2: ArrayLike, rho: ArrayLike) -> np.ndarray:
    """sigma^2 * (1 + rho) / (1 - rho)."""
    rho = np.asarray(rho, dtype=float)
    return np.asarray(sigma2, dtype=float) * (1.0 + rho) / (1.0 - rho)


def ar1_information_factor(rho: ArrayLike) -> np.ndarray:
    """Information per sample after prewhitening, relative to i.i.d.: (1 - rho) / (1 + rho)."""
    rho = np.asarray(rho, dtype=float)
    return (1.0 - rho) / (1.0 + rho)


def lag1_autocorrelation(x: ArrayLike) -> np.ndarray:
    """Lag-1 sample autocorrelation along axis 0 (Yule-Walker estimate of rho)."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean(axis=0)
    return np.sum(xc[1:] * xc[:-1], axis=0) / np.sum(xc * xc, axis=0)


def batch_means_lrv(x: ArrayLike, batch_size: int | None = None) -> np.ndarray:
    """Batch-means estimate of the long-run variance along axis 0.

    batch_size * Var(batch means) over non-overlapping batches; the default
    batch size is floor(sqrt(n)). Trailing samples that do not fill a batch
    are dropped.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    b = int(np.sqrt(n)) if batch_size is None else int(batch_size)
    k = n // b
    if k < 2:
        raise ValueError("need at least two full batches")
    means = x[: k * b].reshape((k, b) + x.shape[1:]).mean(axis=1)
    return b * means.var(axis=0, ddof=1)
