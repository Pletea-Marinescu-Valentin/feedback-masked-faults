"""PCA monitoring with Hotelling's T^2 and the squared prediction error (SPE).

Classical limits for new observations: T^2 from the F distribution
(MacGregor and Kourti, 1995) and SPE from Jackson and Mudholkar (1979).
Both assume independent Gaussian samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass(frozen=True)
class PCAMonitor:
    mean: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray  # p x k
    eigenvalues: np.ndarray  # all p, descending
    n: int

    @property
    def k(self) -> int:
        return self.loadings.shape[1]

    @classmethod
    def fit(cls, x: ArrayLike, variance: float = 0.9) -> "PCAMonitor":
        """Standardize the calibration data and keep the components explaining `variance`."""
        x = np.asarray(x, dtype=float)
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
        scale = np.where(scale > 0, scale, 1.0)
        z = (x - mean) / scale
        eigval, eigvec = np.linalg.eigh(np.cov(z, rowvar=False))
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = np.clip(eigval[order], 0.0, None), eigvec[:, order]
        k = int(np.searchsorted(np.cumsum(eigval) / eigval.sum(), variance) + 1)
        return cls(mean, scale, eigvec[:, :k], eigval, x.shape[0])

    def statistics(self, x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        z = (np.asarray(x, dtype=float) - self.mean) / self.scale
        t = z @ self.loadings
        t2 = np.sum(t**2 / self.eigenvalues[: self.k], axis=-1)
        residual = z - t @ self.loadings.T
        return t2, np.sum(residual**2, axis=-1)

    def limits(self, alpha: float) -> tuple[float, float]:
        n, k = self.n, self.k
        t2 = k * (n**2 - 1) / (n * (n - k)) * stats.f.ppf(1 - alpha, k, n - k)
        rest = self.eigenvalues[k:]
        th1, th2, th3 = (np.sum(rest**i) for i in (1, 2, 3))
        h0 = 1.0 - 2.0 * th1 * th3 / (3.0 * th2**2)
        c = stats.norm.ppf(1 - alpha)
        spe = th1 * (c * np.sqrt(2.0 * th2 * h0**2) / th1 + 1.0
                     + th2 * h0 * (h0 - 1.0) / th1**2) ** (1.0 / h0)
        return float(t2), float(spe)
