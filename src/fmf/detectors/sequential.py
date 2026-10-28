"""Sequential detector statistics on standardized scores, and alarm counting with restarts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from fmf.detectors.conformal import conformal_pvalues, dkw_adjust
from fmf.detectors.cusum import cusum, gaussian_llr
from fmf.detectors.e_detector import log_e_cusum_unfloored, log_mixture, power_calibrator
from fmf.metrics.run_length import first_crossing


def gaussian_cusum_statistic(z: ArrayLike, shift: float) -> np.ndarray:
    """Page's CUSUM on N(0, 1) scores tuned to a mean shift `shift` (NaN: no observation)."""
    return cusum(gaussian_llr(z, shift))


def conformal_e_statistic(scores: ArrayLike, calibration: ArrayLike, kappas: ArrayLike,
                          dkw_delta: float | None) -> np.ndarray:
    """log of the kappa-grid mixture of e-CUSUM statistics on conformal p-values.

    Large scores are evidence of a fault (one-sided). NaN scores are missing
    observations (e = 1). With dkw_delta set, p-values receive the DKW
    correction for the calibration set size.
    """
    scores = np.asarray(scores, dtype=float)
    calibration = np.asarray(calibration, dtype=float)
    calibration = calibration[np.isfinite(calibration)]
    missing = np.isnan(scores)
    p = conformal_pvalues(calibration, np.where(missing, 0.0, scores))
    if dkw_delta is not None:
        p = dkw_adjust(p, calibration.size, dkw_delta)
    stats = []
    for kappa in np.atleast_1d(kappas):
        log_e = np.where(missing, np.nan, np.log(power_calibrator(p, kappa)))
        stats.append(log_e_cusum_unfloored(log_e))
    return log_mixture(np.stack(stats, axis=-1))


def count_alarms(statistic: Callable[[np.ndarray], np.ndarray], x: ArrayLike,
                 threshold: float) -> list[int]:
    """Alarm indices on a 1-D stream when the detector restarts after every alarm."""
    x = np.asarray(x, dtype=float)
    alarms, start = [], 0
    while start < x.size:
        idx = int(first_crossing(statistic(x[start:]), threshold))
        if idx < 0:
            break
        alarms.append(start + idx)
        start += idx + 1
    return alarms
