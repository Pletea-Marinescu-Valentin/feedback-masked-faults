"""Prewhitening of autocorrelated residuals.

An AR(p) model fitted on fault-free calibration residuals turns the residual
into innovations w_t = x_t - sum_i phi_i x_{t-i}. A persistent mean shift mu
in x appears as (1 - sum_i phi_i) * mu in w, and with innovation variance
sigma_w^2 the information per sample is mu^2 (1 - sum phi)^2 / (2 sigma_w^2)
= mu^2 / (2 lrv), lrv = sigma_w^2 / (1 - sum phi)^2 being the long-run
variance of the AR model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def ar1_prewhiten(x: ArrayLike, rho: float) -> np.ndarray:
    """Innovations w_t = x_t - rho * x_{t-1} along axis 0; the first sample is dropped.

    A persistent mean shift mu in x appears as (1 - rho) * mu in w, and
    Var(w) = (1 - rho^2) * Var(x) for a stationary AR(1) x.
    """
    x = np.asarray(x, dtype=float)
    return x[1:] - rho * x[:-1]


@dataclass(frozen=True)
class ARModel:
    phi: np.ndarray  # coefficients for lags 1..p
    sigma_w: float  # innovation standard deviation
    mean: float  # calibration mean removed before filtering

    @property
    def order(self) -> int:
        return self.phi.size

    @property
    def gain(self) -> float:
        """1 - sum(phi): factor applied to a persistent mean shift."""
        return float(1.0 - self.phi.sum())

    @property
    def long_run_variance(self) -> float:
        return self.sigma_w**2 / self.gain**2


def _lagged(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Targets x_t and regressors (x_{t-1}, ..., x_{t-p}) along axis 0."""
    n = x.shape[0]
    lags = np.stack([x[p - i: n - i] for i in range(1, p + 1)], axis=-1)
    return x[p:], lags


def fit_ar(x: ArrayLike, order: int) -> ARModel:
    """Least-squares AR(order) fit on calibration residuals along axis 0.

    x may be (n,) or (n, r); columns are independent series sharing one
    model. Rows with a NaN in the target or any lag are skipped, so gated
    samples (NaN) break the series without bridging the gap.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    mean = float(np.nanmean(x))
    y, lags = _lagged(x - mean, order)
    y = y.reshape(-1)
    lags = lags.reshape(-1, order)
    ok = np.isfinite(y) & np.all(np.isfinite(lags), axis=1)
    phi, *_ = np.linalg.lstsq(lags[ok], y[ok], rcond=None)
    resid = y[ok] - lags[ok] @ phi
    return ARModel(phi=phi, sigma_w=float(resid.std(ddof=order)), mean=mean)


def ar_prewhiten(x: ArrayLike, model: ARModel) -> np.ndarray:
    """Innovations of x under the AR model, along axis 0, same length as x.

    The first `order` samples, and every sample whose current value or one of
    its lags is NaN, are NaN (no observation).
    """
    x = np.asarray(x, dtype=float) - model.mean
    w = np.full_like(x, np.nan)
    p = model.order
    w[p:] = x[p:] - np.tensordot(_lagged(x, p)[1], model.phi, axes=([-1], [0]))
    return w
