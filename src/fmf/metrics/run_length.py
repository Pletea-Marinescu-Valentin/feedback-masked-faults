"""Alarm times, run lengths and detection delays from detector statistics.

Indices are 0-based sample indices along axis 0. An alarm at index i means
i + 1 samples were observed; a change at index nu means sample nu is the
first post-change sample, so an alarm at nu has delay 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def first_crossing(statistic: ArrayLike, threshold: float) -> np.ndarray:
    """First index with statistic >= threshold along axis 0; -1 where never."""
    hit = np.asarray(statistic) >= threshold
    return np.where(hit.any(axis=0), hit.argmax(axis=0), -1)


def run_lengths(alarm_index: ArrayLike, horizon: int) -> np.ndarray:
    """Samples until alarm; runs without alarm are censored at horizon."""
    idx = np.asarray(alarm_index)
    return np.where(idx >= 0, idx + 1, horizon)


def detection_delays(alarm_index: ArrayLike, onset: int) -> np.ndarray:
    """Post-change samples until alarm; NaN for no alarm or an alarm before onset."""
    idx = np.asarray(alarm_index)
    return np.where(idx >= onset, idx - onset + 1.0, np.nan)
