"""Batch means of control-effort residuals with context-only gating.

Averaging the residual over batches much longer than its correlation time
gives nearly independent batch means whose variance is lrv / m, so a detector
on batch means keeps the information rate mu^2 / (2 lrv) per sample while
dropping the need for a model of the residual autocorrelation. The gate must
depend on the context only (e.g. the fault-free command lies in the
modulating range), never on the measured command, so that it selects the same
samples before and after a fault.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def batch_means(x: ArrayLike, size: int, gate: ArrayLike | None = None,
                min_fill: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Means of gated samples over consecutive batches of `size` samples along axis 0.

    Returns (means, counts), each of shape (n // size,) + x.shape[1:]; batches
    with fewer than min_fill * size gated samples have mean NaN and count 0.
    """
    x = np.asarray(x, dtype=float)
    k = x.shape[0] // size
    xb = x[: k * size].reshape((k, size) + x.shape[1:])
    if gate is None:
        g = np.ones(xb.shape, dtype=bool)
    else:
        g = np.asarray(gate, dtype=bool)[: k * size]
        g = np.broadcast_to(g.reshape((k, size) + g.shape[1:] + (1,) * (x.ndim - g.ndim)),
                            xb.shape)
    counts = g.sum(axis=1)
    sums = np.where(g, xb, 0.0).sum(axis=1)
    enough = counts >= min_fill * size
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(enough, sums / counts, np.nan)
    return means, np.where(enough, counts, 0)


def standardize(means: ArrayLike, counts: ArrayLike, lrv: ArrayLike) -> np.ndarray:
    """z = mean * sqrt(count / lrv): N(0, 1) under H0 when lrv is the residual's long-run variance."""
    return np.asarray(means, dtype=float) * np.sqrt(np.asarray(counts, dtype=float)
                                                    / np.asarray(lrv, dtype=float))
