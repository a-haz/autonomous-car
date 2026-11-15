"""GT/INS stub backend (PR-06a).

Passes the simulator's ground-truth pose through, optionally with seeded
Gaussian noise. The reported covariance is the noise actually added, so
sqrt(P_xx + P_yy) is honest. With all sigmas at 0 the output equals GT.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .pose import HealthMonitor, PoseEstimate, pose_covariance

GtPose = Callable[[], tuple[float, float, float, float]]  # x, y, yaw, v


def _wrap(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class LocBackend(Protocol):
    def estimate(self, stamp_ns: int) -> PoseEstimate: ...


class GtInsStub:
    def __init__(self, gt: GtPose, sigma_xy: float = 0.0, sigma_yaw: float = 0.0,
                 sigma_v: float = 0.0, seed: int | None = None) -> None:
        if min(sigma_xy, sigma_yaw, sigma_v) < 0.0:
            raise ValueError("sigma must be >= 0")
        self.gt = gt
        self.sigma_xy, self.sigma_yaw, self.sigma_v = sigma_xy, sigma_yaw, sigma_v
        self._rng = np.random.default_rng(seed)
        self._cov = pose_covariance(sigma_xy**2, sigma_xy**2, sigma_yaw**2)

    def estimate(self, stamp_ns: int) -> PoseEstimate:
        x, y, yaw, v = self.gt()
        if self.sigma_xy > 0.0 or self.sigma_yaw > 0.0 or self.sigma_v > 0.0:
            n = self._rng.standard_normal(4)
            x += self.sigma_xy * n[0]
            y += self.sigma_xy * n[1]
            yaw = _wrap(yaw + self.sigma_yaw * n[2])
            v += self.sigma_v * n[3]
        return PoseEstimate(stamp_ns, float(x), float(y), float(yaw), float(v),
                            self._cov.copy(), self.sigma_v**2)


class Localization:
    """Always-on front end: one backend in, latest /localization/pose out."""

    def __init__(self, backend: LocBackend, health: HealthMonitor | None = None) -> None:
        self.backend = backend
        self.health = health or HealthMonitor()
        self.pose: PoseEstimate | None = None
        self.healthy = True

    def step(self, stamp_ns: int) -> PoseEstimate:
        self.pose = self.backend.estimate(stamp_ns)
        self.healthy = self.health.update(self.pose)
        return self.pose
