"""Detection-delay predictions for a mean shift in the control-effort residual (C1).

A Gaussian mean shift mu with variance sigma^2 carries I = mu^2 / (2 sigma^2)
nats per sample. Page's CUSUM tuned to the shift has worst-case delay
D ~ ln(ARL0) / I as ARL0 -> infinity (Lorden 1971). At finite ARL0 the
Siegmund approximation for Gaussian increments is used instead:

    ARL(drift, b) ~ (exp(-2 drift b') + 2 drift b' - 1) / (2 drift^2),  b' = b + 1.166,

for the standardized CUSUM max(0, S + X - k), X - k ~ N(drift, 1), threshold b.
Its expansions give ln ARL0 = h + 1.166 * delta + ln(2 / delta^2) + o(1) and
ARL1 = (h + 1.166 * delta - 1) / I + o(1) on the log-likelihood-ratio scale
(delta = mu / sigma), so the delay is linear in ln ARL0 with slope 1 / I but
offset by -(ln(2 / delta^2) + 1) / I. Ratios of delays across noise levels
therefore retain finite-ARL0 terms and should be compared through the
corrected formula, not through 1 / I alone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

SIEGMUND_OFFSET = 1.166  # 2 * 0.583: expected-overshoot correction for Gaussian increments


def gaussian_information(mu: ArrayLike, sigma: ArrayLike) -> np.ndarray:
    """Kullback-Leibler divergence per sample of N(mu, sigma^2) from N(0, sigma^2)."""
    return np.asarray(mu, dtype=float) ** 2 / (2.0 * np.asarray(sigma, dtype=float) ** 2)


def asymptotic_delay(arl0: ArrayLike, information: ArrayLike) -> np.ndarray:
    """ln(ARL0) / I."""
    return np.log(np.asarray(arl0, dtype=float)) / np.asarray(information, dtype=float)


def siegmund_arl(drift: float, b: float) -> float:
    """Siegmund's ARL approximation for the standardized Gaussian CUSUM."""
    bp = b + SIEGMUND_OFFSET
    x = 2.0 * drift * bp
    if abs(x) < 1e-6:
        return bp**2
    return float((np.expm1(-x) + x) / (2.0 * drift**2))


def gaussian_cusum_arl(h: float, shift: float, sigma: float = 1.0,
                       actual_shift: float = 0.0) -> float:
    """ARL of the CUSUM on increments (shift / sigma^2) * (x - shift / 2), threshold h.

    actual_shift is the true mean of x: 0 gives ARL0, shift gives the
    zero-state delay under the design alternative.
    """
    delta = shift / sigma
    drift = actual_shift / sigma - delta / 2.0
    return siegmund_arl(drift, h / delta)


def gaussian_cusum_threshold(arl0: float, shift: float, sigma: float = 1.0) -> float:
    """Threshold h whose Siegmund ARL0 equals arl0."""
    def gap(h: float) -> float:
        return np.log(gaussian_cusum_arl(h, shift, sigma)) - np.log(arl0)

    if gap(0.0) >= 0.0:
        raise ValueError("arl0 is below the ARL0 of a zero threshold")
    return brentq(gap, 0.0, np.log(arl0) + 20.0)


def brownian_cusum_delay(information: float, arl0: float) -> float:
    """Delay of the matched CUSUM in the Brownian limit (small per-sample information).

    With log-likelihood-ratio drift +-I and variance 2I per sample,
    ARL0 = (e^h - h - 1) / I and D = (e^-h + h - 1) / I, hence
    D = (ln(I * ARL0) - 1) / I + o(1): the asymptotic ln(ARL0) / I overstates
    the delay by ln(1 / I) / I. With I = i * dt and ARL0 = A / dt the delay in
    time units, D * dt = (ln(i * A) - 1) / i, does not depend on the sampling
    period.
    """
    target = information * arl0

    def gap(h: float) -> float:
        return np.expm1(h) - h - target

    if target <= 0.0:
        raise ValueError("information * arl0 must be positive")
    h = brentq(gap, 0.0, np.log(target + 1.0) + 2.0)
    return float((np.expm1(-h) + h) / information)


def e_detector_min_delay(n: int, alpha: float, kappas: ArrayLike,
                         dkw_delta: float | None = None) -> float:
    """Fewest observations before a kappa-mixture e-CUSUM can alarm, for any fault size.

    With n calibration scores the conformal p-value is at least 1 / (n + 1),
    plus eps = sqrt(ln(1 / delta) / (2 n)) under the DKW correction, so each
    e-value is at most c = kappa * p_min^(kappa - 1). Each e-CUSUM grows with
    every e-value, so it is largest when all of them equal c, where
    M_t = max(c, c^t); the mixture alarms at the first t at which the average of
    these over the K kappas reaches 1 / alpha. Scores above every calibration
    score attain this delay. It lies between ln(1 / alpha) / ln c_max and
    ln(K / alpha) / ln c_max.
    """
    kappas = np.atleast_1d(np.asarray(kappas, dtype=float))
    p_min = 1.0 / (n + 1.0)
    if dkw_delta is not None:
        p_min += np.sqrt(np.log(1.0 / dkw_delta) / (2.0 * n))
    log_c = np.log(kappas) + (kappas - 1.0) * np.log(p_min)
    best = log_c.max()
    if best <= 0.0:
        return float("inf")
    target = np.log(kappas.size / alpha)
    first = max(1, int(np.ceil(np.log(1.0 / alpha) / best)))
    last = int(np.ceil(target / best))
    for t in range(first, last):
        if np.logaddexp.reduce(np.maximum(log_c, t * log_c)) >= target:
            return float(t)
    return float(last)


def cumulative_information_delay(information: ArrayLike, arl0: float) -> np.ndarray:
    """First index t with sum_{s <= t} I_s >= ln(ARL0), along axis 0; -1 if never.

    First-order delay of a CUSUM whose log-likelihood ratio follows the
    time-varying signature mu_t: windows with I_t = 0 pause the accumulation.
    A CUSUM tuned to a fixed shift mu_1 instead drifts down at rate
    (mu_1 / sigma^2) * (mu_1 / 2 - mu_t) where mu_t < mu_1 / 2, so such
    windows erase evidence rather than pause it.
    """
    cum = np.cumsum(np.asarray(information, dtype=float), axis=0)
    hit = cum >= np.log(arl0)
    return np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
