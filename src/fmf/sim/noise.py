"""Noise sources for the synthetic loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class NoiseModel:
    """Measurement noise and unmeasured disturbance acting on the CV.

    sensor_std: white noise on the CV sensor [K].
    quantization: CV sensor resolution [K]; 0 disables quantization.
    disturbance_std: stationary std of the unmeasured disturbance added to the
        open-loop CV (e.g. coil water temperature or latent load) [K].
    disturbance_rho: AR(1) coefficient of that disturbance per sample.
    """

    sensor_std: float = 0.0
    quantization: float = 0.0
    disturbance_std: float = 0.0
    disturbance_rho: float = 0.0


def ar1(n: int, rho: float, std: float, size: int | None = None,
        rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Stationary zero-mean AR(1): x_t = rho * x_{t-1} + sqrt(1 - rho^2) * std * eps_t.

    Returns shape (n,) if size is None, else (n, size).
    """
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie in (-1, 1)")
    rng = np.random.default_rng(rng)
    eps = rng.standard_normal((n,) if size is None else (n, size))
    x = np.empty_like(eps)
    x[0] = std * eps[0]
    if n > 1:
        zi = np.asarray(rho * x[0])[np.newaxis, ...]
        x[1:], _ = lfilter([std * np.sqrt(1.0 - rho**2)], [1.0, -rho], eps[1:],
                           axis=0, zi=zi)
    return x
