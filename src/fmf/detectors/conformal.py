"""Conformal p-values against a fault-free calibration set, with a DKW correction.

p_t = (1 + #{i : s_i >= s_t}) / (n + 1) is super-uniform marginally, i.e. over
the joint draw of calibration and test scores. Conditionally on the
calibration set it is not, and a sequential detector reuses that set for every
test point. With eps = sqrt(ln(1 / delta) / (2 n)), Massart's one-sided DKW
bound (valid for delta <= 1/2) gives, with probability >= 1 - delta over the
calibration draw, S_hat(s) >= S(s) - eps for all s, where S is the survival
function of the score and S_hat its empirical version. On that event
p_t >= S_hat(s_t) >= S(s_t) - eps, so p~_t = min(1, p_t + eps) >= S(s_t) is
super-uniform conditionally on the calibration set. Given that set, p-values
of i.i.d. test scores are i.i.d., which is what the e-detector needs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def conformal_pvalues(calibration_scores: ArrayLike, scores: ArrayLike) -> np.ndarray:
    """(1 + #{i : s_i >= s}) / (n + 1) for each test score s; large scores give small p."""
    cal = np.sort(np.asarray(calibration_scores, dtype=float).ravel())
    n = cal.size
    n_ge = n - np.searchsorted(cal, np.asarray(scores, dtype=float), side="left")
    return (1.0 + n_ge) / (n + 1.0)


def dkw_epsilon(n: int, delta: float) -> float:
    """sqrt(ln(1 / delta) / (2 n)), Massart's one-sided bound; requires delta <= 1/2."""
    if not 0.0 < delta <= 0.5:
        raise ValueError("delta must lie in (0, 1/2]")
    return float(np.sqrt(np.log(1.0 / delta) / (2.0 * n)))


def dkw_adjust(p: ArrayLike, n: int, delta: float) -> np.ndarray:
    """min(1, p + eps): calibration-conditionally valid with probability >= 1 - delta."""
    return np.minimum(1.0, np.asarray(p, dtype=float) + dkw_epsilon(n, delta))
