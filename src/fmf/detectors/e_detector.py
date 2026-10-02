"""p-to-e calibration and e-detectors for sequential change detection.

Calibrators (Vovk and Wang 2021) map a super-uniform p to an e-value with
E[e] <= 1. With e-values satisfying E[e_t | F_{t-1}] <= 1 before the change,
the e-detectors of Shin, Ramdas and Rinaldo raise an alarm at the first t with
statistic >= 1 / alpha and guarantee ARL0 >= 1 / alpha:

    e-CUSUM:  M_t = e_t * max(M_{t-1}, 1),
    e-SR:     R_t = e_t * (1 + R_{t-1}),   R_0 = 0.

R_t - t is a supermartingale, so E[tau_SR] >= E[R_tau] >= 1 / alpha, and
M_t <= R_t makes the e-CUSUM alarm no earlier than e-SR. Both are computed in
log space: max(log M_t, 0) is Page's CUSUM on log e_t, so the e-CUSUM alarm
is that CUSUM crossing ln(1 / alpha).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from fmf.detectors.cusum import cusum


def power_calibrator(p: ArrayLike, kappa: float) -> np.ndarray:
    """kappa * p^(kappa - 1), kappa in (0, 1)."""
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie in (0, 1)")
    return kappa * np.asarray(p, dtype=float) ** (kappa - 1.0)


def mixture_calibrator(p: ArrayLike) -> np.ndarray:
    """Uniform mixture of power calibrators over kappa in (0, 1).

    int_0^1 kappa p^(kappa - 1) d kappa = (1 - p + p ln p) / (p ln^2 p), with value 1/2 at p = 1.
    """
    p = np.asarray(p, dtype=float)
    lp = np.log(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (1.0 - p + p * lp) / (p * lp**2)
    return np.where(p >= 1.0, 0.5, e)


def log_e_cusum(log_e: ArrayLike) -> np.ndarray:
    """max(log M_t, 0) along axis 0; the alarm is the first crossing of ln(1 / alpha)."""
    return cusum(log_e)


def log_e_shiryaev_roberts(log_e: ArrayLike) -> np.ndarray:
    """log R_t along axis 0.

    R_t = sum_{k <= t} prod_{j=k}^{t} e_j, so log R_t = S_t + log sum_{k <= t} exp(-S_{k-1})
    with S the partial sums of log e and S_0 = 0.
    """
    log_e = np.asarray(log_e, dtype=float)
    s = np.cumsum(log_e, axis=0)
    s_prev = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
    return s + np.logaddexp.accumulate(-s_prev, axis=0)
