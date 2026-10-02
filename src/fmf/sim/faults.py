"""Fault trajectories on the sample grid."""

from __future__ import annotations

import numpy as np


def step_profile(n: int, onset: int, magnitude: float) -> np.ndarray:
    """0 before sample `onset`, `magnitude` from it on."""
    x = np.zeros(n)
    x[onset:] = magnitude
    return x


def ramp_profile(n: int, onset: int, duration: int, magnitude: float) -> np.ndarray:
    """Linear rise from 0 at `onset` to `magnitude` after `duration` samples, then constant."""
    if duration <= 0:
        return step_profile(n, onset, magnitude)
    k = np.arange(n) - onset
    return magnitude * np.clip(k / duration, 0.0, 1.0)
