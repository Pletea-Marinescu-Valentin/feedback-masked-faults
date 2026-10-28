"""Synthetic closed loop (PI + first-order plant) with feedback-masked faults."""

from fmf.sim.context import Context, constant_context, seasonal_context
from fmf.sim.faults import ramp_profile, step_profile
from fmf.sim.loop import LoopParams, LoopResult, simulate
from fmf.sim.noise import NoiseModel, ar1

__all__ = [
    "Context",
    "LoopParams",
    "LoopResult",
    "NoiseModel",
    "ar1",
    "constant_context",
    "ramp_profile",
    "seasonal_context",
    "simulate",
    "step_profile",
]
