"""Normalized valve characteristics phi: [0, 1] -> [0, 1], strictly increasing.

The delivered capacity is q = theta * phi(u), with theta the capacity at full
opening and u the actuator command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class Valve(Protocol):
    def phi(self, u: ArrayLike) -> np.ndarray: ...

    def phi_inv(self, q: ArrayLike) -> np.ndarray: ...

    def dphi(self, u: ArrayLike) -> np.ndarray: ...


@dataclass(frozen=True)
class LinearValve:
    """phi(u) = u."""

    def phi(self, u: ArrayLike) -> np.ndarray:
        return np.asarray(u, dtype=float)

    def phi_inv(self, q: ArrayLike) -> np.ndarray:
        return np.asarray(q, dtype=float)

    def dphi(self, u: ArrayLike) -> np.ndarray:
        return np.ones_like(np.asarray(u, dtype=float))


@dataclass(frozen=True)
class EqualPercentageValve:
    """phi(u) = (R**u - 1) / (R - 1), the equal-percentage law shifted to phi(0) = 0."""

    rangeability: float = 50.0

    def __post_init__(self) -> None:
        if self.rangeability <= 1.0:
            raise ValueError("rangeability must exceed 1")

    def phi(self, u: ArrayLike) -> np.ndarray:
        r = self.rangeability
        return np.expm1(np.asarray(u, dtype=float) * np.log(r)) / (r - 1.0)

    def phi_inv(self, q: ArrayLike) -> np.ndarray:
        r = self.rangeability
        return np.log1p(np.asarray(q, dtype=float) * (r - 1.0)) / np.log(r)

    def dphi(self, u: ArrayLike) -> np.ndarray:
        r = self.rangeability
        return np.log(r) * np.exp(np.asarray(u, dtype=float) * np.log(r)) / (r - 1.0)
