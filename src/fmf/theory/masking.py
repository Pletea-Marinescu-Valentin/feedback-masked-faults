"""Quasi-static relations of a SISO loop with integral action (contribution C1).

Delivered capacity q = theta * phi(u). With sign = -1 for an actuator that
lowers the controlled variable (CV, e.g. a cooling coil on supply-air
temperature) and sign = +1 for one that raises it, the steady-state true CV is

    y = y_free + sign * q / (mdot * cp),

where y_free is the CV with the actuator closed. The load L is the capacity
that holds the true CV at the setpoint r:

    L = sign * mdot * cp * (r - y_free).

Unsaturated, the integrator drives the measured CV y_m to r, so
u* = phi^{-1}(L / theta). A CV sensor bias b (y_m = y + b) moves the true CV
to r - b and the required capacity to L - sign * mdot * cp * b; a capacity loss
theta = theta0 * (1 - delta) leaves the CV at r and raises u. In both cases
the measured CV carries no steady-state trace of the fault until the actuator
saturates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from fmf.valves import Valve

CP_AIR = 1.006  # specific heat of air [kJ/(kg K)]


def required_load(y_free: ArrayLike, setpoint: ArrayLike, mdot: ArrayLike,
                  cp: float = CP_AIR, sign: int = -1) -> np.ndarray:
    """Capacity L [kW] that holds the true CV at the setpoint."""
    return sign * np.asarray(mdot, dtype=float) * cp * (
        np.asarray(setpoint, dtype=float) - np.asarray(y_free, dtype=float))


def biased_load(load: ArrayLike, bias: ArrayLike, mdot: ArrayLike,
                cp: float = CP_AIR, sign: int = -1) -> np.ndarray:
    """Capacity that holds the measured CV at the setpoint under a CV sensor bias."""
    return (np.asarray(load, dtype=float)
            - sign * np.asarray(mdot, dtype=float) * cp * np.asarray(bias, dtype=float))


def steady_state_command(load: ArrayLike, theta: ArrayLike, valve: Valve,
                         u_min: float = 0.0, u_max: float = 1.0) -> np.ndarray:
    """u* = phi^{-1}(L / theta), clipped to the actuator range."""
    frac = np.clip(np.asarray(load, dtype=float) / np.asarray(theta, dtype=float),
                   valve.phi(u_min), valve.phi(u_max))
    return valve.phi_inv(frac)


def bias_command_shift(bias: ArrayLike, load: ArrayLike, theta: ArrayLike,
                       mdot: ArrayLike, valve: Valve, cp: float = CP_AIR,
                       sign: int = -1, u_min: float = 0.0,
                       u_max: float = 1.0) -> np.ndarray:
    """Steady-state shift of u caused by a CV sensor bias.

    Unsaturated with linear phi: Delta u = -sign * mdot * cp * b / theta,
    i.e. mdot * cp * b / theta for a cooling coil.
    """
    biased = biased_load(load, bias, mdot, cp, sign)
    return (steady_state_command(biased, theta, valve, u_min, u_max)
            - steady_state_command(load, theta, valve, u_min, u_max))


def fouling_command_shift(delta: ArrayLike, load: ArrayLike, theta0: float,
                          valve: Valve, u_min: float = 0.0,
                          u_max: float = 1.0) -> np.ndarray:
    """Steady-state shift of u caused by a capacity loss theta = theta0 * (1 - delta).

    Unsaturated with linear phi: Delta u = u_n * delta / (1 - delta) with
    u_n = L / theta0. It vanishes with the load: no information while the
    actuator rests.
    """
    theta = theta0 * (1.0 - np.asarray(delta, dtype=float))
    return (steady_state_command(load, theta, valve, u_min, u_max)
            - steady_state_command(load, theta0, valve, u_min, u_max))


def measured_cv_offset(load: ArrayLike, theta: ArrayLike, mdot: ArrayLike,
                       valve: Valve, bias: ArrayLike = 0.0, cp: float = CP_AIR,
                       sign: int = -1, u_min: float = 0.0,
                       u_max: float = 1.0) -> np.ndarray:
    """Steady-state y_m - r: zero while unsaturated, nonzero once u saturates."""
    biased = biased_load(load, bias, mdot, cp, sign)
    u = steady_state_command(biased, theta, valve, u_min, u_max)
    return -sign * (biased - np.asarray(theta, dtype=float) * valve.phi(u)) / (
        np.asarray(mdot, dtype=float) * cp)


def critical_capacity(design_load: ArrayLike, valve: Valve,
                      u_max: float = 1.0) -> np.ndarray:
    """theta* = L* / phi(u_max): below it the design load saturates the actuator."""
    return np.asarray(design_load, dtype=float) / valve.phi(u_max)


def plant_dc_gain(theta: ArrayLike, mdot: ArrayLike, u: ArrayLike, valve: Valve,
                  cp: float = CP_AIR, sign: int = -1) -> np.ndarray:
    """Static sensitivity P(0) = dy/du = sign * theta * phi'(u) / (mdot * cp) [K]."""
    return sign * np.asarray(theta, dtype=float) * valve.dphi(u) / (
        np.asarray(mdot, dtype=float) * cp)


def command_long_run_variance(cv_lrv: ArrayLike, dc_gain: ArrayLike) -> np.ndarray:
    """Long-run variance of u induced by CV measurement noise or output disturbances.

    With integral action the closed-loop map from such an input to u has DC
    gain -1 / P(0) for any stabilizing PI tuning, so the zero-frequency
    variance of u is lrv / P(0)^2.
    """
    return np.asarray(cv_lrv, dtype=float) / np.asarray(dc_gain, dtype=float) ** 2
