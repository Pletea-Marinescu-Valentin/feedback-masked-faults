"""Synthetic SISO loop: first-order plant, velocity-form PI, CV sensor and capacity faults.

At sample t the controller reads y_m[t] = y[t] + b[t] + v[t] (quantized),
computes u[t], and holds it until t + 1. The plant is a first-order lag,
discretized exactly under zero-order hold:

    y[t+1] = a * y[t] + (1 - a) * (y_free[t] + sign * theta[t] * phi(u[t]) / (mdot[t] * cp)),

with a = exp(-dt / tau), y_free = t_in + d (d: unmeasured AR(1) disturbance)
and theta[t] = theta0 * (1 - delta[t]).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fmf.sim.context import Context
from fmf.sim.noise import NoiseModel, ar1
from fmf.theory.masking import CP_AIR, biased_load, steady_state_command
from fmf.valves import LinearValve, Valve


@dataclass(frozen=True)
class LoopParams:
    """Plant and controller parameters.

    Defaults model an AHU cooling coil on supply-air temperature. The PI gains
    follow lambda tuning at the nominal point (linear valve, mdot = 5 kg/s):
    |P(0)| = theta0 / (mdot * cp) = 19.9 K, lambda = 180 s,
    kp = tau / (|P(0)| * lambda), ti = tau.
    """

    theta0: float = 100.0  # capacity at full opening [kW]
    tau: float = 120.0  # plant time constant [s]
    kp: float = 0.0335  # proportional gain [1/K]
    ti: float = 120.0  # integral time [s]
    u_min: float = 0.0
    u_max: float = 1.0
    sign: int = -1  # -1: actuator lowers the CV (cooling); +1: raises it (heating)
    cp: float = CP_AIR  # [kJ/(kg K)]
    valve: Valve = field(default_factory=LinearValve)


@dataclass(frozen=True)
class LoopResult:
    """Sampled trajectories: (n,) for a single run, (n, n_replicas) otherwise."""

    context: Context
    params: LoopParams
    y: np.ndarray  # true CV [degC]
    y_meas: np.ndarray  # measured CV [degC]
    u: np.ndarray  # actuator command
    bias: np.ndarray  # CV sensor bias [K], (n,)
    fouling: np.ndarray  # capacity loss fraction, (n,)

    @property
    def t(self) -> np.ndarray:
        return self.context.t

    @property
    def load(self) -> np.ndarray:
        """Load from the measured context [kW], (n,)."""
        return self.context.load(self.params.cp, self.params.sign)

    @property
    def theta(self) -> np.ndarray:
        """Actual capacity at full opening [kW], (n,)."""
        return self.params.theta0 * (1.0 - self.fouling)

    @property
    def u_nominal(self) -> np.ndarray:
        """Fault-free quasi-static command phi^{-1}(L(z) / theta0): the oracle u_hat(z), (n,)."""
        p = self.params
        return steady_state_command(self.load, p.theta0, p.valve, p.u_min, p.u_max)

    @property
    def residual(self) -> np.ndarray:
        """u - u_hat(z) with the oracle baseline."""
        u_hat = self.u_nominal
        return self.u - (u_hat[:, np.newaxis] if self.u.ndim == 2 else u_hat)

    def to_frame(self, replica: int = 0) -> pd.DataFrame:
        def pick(x: np.ndarray) -> np.ndarray:
            return x[:, replica] if x.ndim == 2 else x

        return pd.DataFrame({
            "t": self.t,
            "t_in": self.context.t_in,
            "mdot": self.context.mdot,
            "setpoint": self.context.setpoint,
            "load": self.load,
            "theta": self.theta,
            "bias": self.bias,
            "fouling": self.fouling,
            "y": pick(self.y),
            "y_meas": pick(self.y_meas),
            "u": pick(self.u),
            "u_nominal": self.u_nominal,
        })


def simulate(params: LoopParams, context: Context, *, bias: ArrayLike = 0.0,
             fouling: ArrayLike = 0.0, noise: NoiseModel | None = None,
             n_replicas: int | None = None,
             rng: np.random.Generator | int | None = None) -> LoopResult:
    """Simulate the closed loop on the context grid.

    bias and fouling are scalars or (n,) trajectories. The run starts at the
    noise-free steady state of sample 0. Replicas share context and faults and
    draw independent noise.
    """
    p = params
    n = context.n
    r = 1 if n_replicas is None else int(n_replicas)
    noise = NoiseModel() if noise is None else noise
    rng = np.random.default_rng(rng)

    bias = np.broadcast_to(np.asarray(bias, dtype=float), (n,)).copy()
    fouling = np.broadcast_to(np.asarray(fouling, dtype=float), (n,)).copy()
    if np.any((fouling < 0.0) | (fouling >= 1.0)):
        raise ValueError("fouling must lie in [0, 1)")

    a = np.exp(-context.dt / p.tau)
    ki = p.kp * context.dt / p.ti
    theta = p.theta0 * (1.0 - fouling)
    span = theta / (context.mdot * p.cp)  # CV change at full opening [K]

    if noise.disturbance_std > 0.0:
        d = ar1(n, noise.disturbance_rho, noise.disturbance_std, size=r, rng=rng)
    else:
        d = np.zeros((n, r))
    if noise.sensor_std > 0.0:
        v = noise.sensor_std * rng.standard_normal((n, r))
    else:
        v = np.zeros((n, r))
    y_free = context.t_in[:, np.newaxis] + d

    load0 = biased_load(context.load(p.cp, p.sign)[0], bias[0], context.mdot[0],
                        p.cp, p.sign)
    u = np.full(r, steady_state_command(load0, theta[0], p.valve, p.u_min, p.u_max))
    y = context.t_in[0] + p.sign * span[0] * p.valve.phi(u)
    e_prev = p.sign * (context.setpoint[0] - (y + bias[0]))

    y_out = np.empty((n, r))
    ym_out = np.empty((n, r))
    u_out = np.empty((n, r))
    q = noise.quantization
    for t in range(n):
        ym = y + bias[t] + v[t]
        if q > 0.0:
            ym = q * np.round(ym / q)
        e = p.sign * (context.setpoint[t] - ym)
        u = np.clip(u + p.kp * (e - e_prev) + ki * e, p.u_min, p.u_max)
        e_prev = e
        y_out[t] = y
        ym_out[t] = ym
        u_out[t] = u
        y = a * y + (1.0 - a) * (y_free[t] + p.sign * span[t] * p.valve.phi(u))

    if n_replicas is None:
        y_out, ym_out, u_out = y_out[:, 0], ym_out[:, 0], u_out[:, 0]
    return LoopResult(context, params, y_out, ym_out, u_out, bias, fouling)
