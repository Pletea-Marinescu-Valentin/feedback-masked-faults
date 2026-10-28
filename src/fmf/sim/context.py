"""Exogenous operating context z_t for the synthetic loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from fmf.theory.masking import CP_AIR, required_load

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class Context:
    """Context sampled every dt seconds.

    t_in: coil inlet air temperature, i.e. the CV with the actuator closed [degC].
    mdot: air mass flow [kg/s].
    setpoint: CV setpoint [degC].
    """

    dt: float
    t_in: np.ndarray
    mdot: np.ndarray
    setpoint: np.ndarray

    def __post_init__(self) -> None:
        n = np.asarray(self.t_in).size
        for name in ("t_in", "mdot", "setpoint"):
            value = np.broadcast_to(np.asarray(getattr(self, name), dtype=float), (n,))
            object.__setattr__(self, name, value.copy())
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if np.any(self.mdot <= 0):
            raise ValueError("mdot must be positive")

    @property
    def n(self) -> int:
        return self.t_in.size

    @property
    def t(self) -> np.ndarray:
        """Sample times [s]."""
        return np.arange(self.n) * self.dt

    def load(self, cp: float = CP_AIR, sign: int = -1) -> np.ndarray:
        """Load L(z) [kW] computed from the measured context."""
        return required_load(self.t_in, self.setpoint, self.mdot, cp, sign)


def constant_context(n: int, dt: float = 60.0, t_in: ArrayLike = 24.0,
                     mdot: ArrayLike = 5.0, setpoint: ArrayLike = 13.0) -> Context:
    """Context with constant (or explicitly given) values on n samples."""
    return Context(dt, np.broadcast_to(np.asarray(t_in, dtype=float), (n,)), mdot,
                   setpoint)


def seasonal_context(n_days: float, dt: float = 60.0, t_in_mean: float = 18.0,
                     seasonal_amplitude: float = 8.0, diurnal_amplitude: float = 3.0,
                     peak_day: float = 200.0, peak_hour: float = 15.0,
                     start_day: float = 0.0, mdot: float = 5.0,
                     setpoint: float = 13.0) -> Context:
    """Inlet temperature with annual and daily cosines.

    Defaults give 7-29 degC against a 13 degC setpoint: the cooling coil idles
    in winter and at night in the shoulder seasons, which produces the
    zero-information windows for capacity faults.
    """
    n = int(round(n_days * SECONDS_PER_DAY / dt))
    day = start_day + np.arange(n) * dt / SECONDS_PER_DAY
    t_in = (t_in_mean
            + seasonal_amplitude * np.cos(2 * np.pi * (day - peak_day) / 365.0)
            + diurnal_amplitude * np.cos(2 * np.pi * (day - peak_hour / 24.0)))
    return Context(dt, t_in, mdot, setpoint)
