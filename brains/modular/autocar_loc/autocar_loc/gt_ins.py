"""GT/INS stub backend (PR-06a, loc_drop hook PR-06b).

Passes the simulator's ground-truth pose through, optionally with seeded
Gaussian noise. The reported covariance is the noise actually added, so
sqrt(P_xx + P_yy) is honest. With all sigmas at 0 the output equals GT.

GNSS dropout (/scenario/inject/loc_drop). While a drop is active the stub
gets no absolute fix. Position is dead reckoned from the INS speed and
heading (the GT v and yaw, with the configured noise), starting from the
last published position. Heading and speed stay INS-aided.

The stub has no IMU or wheel-odometry error model of its own yet (that
comes with the EKF), so the position covariance during a drop is a
conservative bound, not an error matched to the dead reckoning above: per
axis, the constant-velocity Kalman prediction with white acceleration noise
of spectral density DR_ACCEL_PSD = A_BOUND**2, where A_BOUND = 3.0 m/s^2 is
the largest acceleration magnitude the plant can command (|A_MIN| in
autocar_ctrl.bicycle). Starting from the last fix covariance, per axis:

    P_pp(t) = P_pp0 + 2 P_pv0 t + P_vv0 t^2 + q t^3 / 3
    P_pv(t) = P_pv0 + P_vv0 t + q t^2 / 2
    P_vv(t) = P_vv0 + q t

with P_vv0 = sigma_v^2. With zero fix noise, sqrt(P_xx + P_yy) crosses the
0.8 m sim bar 0.474 s after the last fix. When the drop ends, the first estimate
is a GNSS fix again and the covariance returns to the fix covariance.

Every estimate outside a drop counts as an absolute fix, so abs_fix_ns is
the stamp of the last estimate before the drop until the drop ends.
"""
from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np

from .pose import HealthMonitor, PoseEstimate, pose_covariance

GtPose = Callable[[], tuple[float, float, float, float]]  # x, y, yaw, v

LOC_DROP_SRV = "/scenario/inject/loc_drop"  # autocar_msgs/srv/InjectLocDrop
A_BOUND = 3.0  # m/s^2, |A_MIN| of the plant (autocar_ctrl.bicycle)
DR_ACCEL_PSD = A_BOUND**2  # (m/s^2)^2 / Hz per axis


def _wrap(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class LocBackend(Protocol):
    def estimate(self, stamp_ns: int) -> PoseEstimate: ...


class GtInsStub:
    def __init__(self, gt: GtPose, sigma_xy: float = 0.0, sigma_yaw: float = 0.0,
                 sigma_v: float = 0.0, seed: int | None = None,
                 dr_accel_psd: float = DR_ACCEL_PSD) -> None:
        if min(sigma_xy, sigma_yaw, sigma_v, dr_accel_psd) < 0.0:
            raise ValueError("sigma must be >= 0")
        self.gt = gt
        self.sigma_xy, self.sigma_yaw, self.sigma_v = sigma_xy, sigma_yaw, sigma_v
        self.dr_accel_psd = dr_accel_psd
        self._rng = np.random.default_rng(seed)
        self._cov = pose_covariance(sigma_xy**2, sigma_xy**2, sigma_yaw**2)
        self._drop_until_ns: int | None = None
        self._last: PoseEstimate | None = None
        self._dr_t0_ns = 0
        self._dr_xy: tuple[float, float] = (0.0, 0.0)
        self._last_fix_ns: int | None = None

    @property
    def gnss_ok(self) -> bool:
        return self._drop_until_ns is None

    def inject_loc_drop(self, duration_s: float, now_ns: int) -> dict:
        """LOC_DROP_SRV handler: no GNSS fix for estimates in [now, now + duration)."""
        if not (math.isfinite(duration_s) and duration_s > 0.0):
            return {"ok": False, "message": "duration_s must be > 0"}
        if self._drop_until_ns is not None:
            return {"ok": False, "message": "loc_drop already active"}
        self._drop_until_ns = now_ns + int(round(duration_s * 1e9))
        return {"ok": True, "message": f"gnss muted for {duration_s:.3f} s"}

    def _sample(self) -> tuple[float, float, float, float]:
        x, y, yaw, v = self.gt()
        if self.sigma_xy > 0.0 or self.sigma_yaw > 0.0 or self.sigma_v > 0.0:
            n = self._rng.standard_normal(4)
            x += self.sigma_xy * n[0]
            y += self.sigma_xy * n[1]
            yaw = _wrap(yaw + self.sigma_yaw * n[2])
            v += self.sigma_v * n[3]
        return float(x), float(y), float(yaw), float(v)

    def _dr_cov(self, t: float) -> np.ndarray:
        p0, vv0, q = self.sigma_xy**2, self.sigma_v**2, self.dr_accel_psd
        var = p0 + vv0 * t * t + q * t**3 / 3.0
        return pose_covariance(var, var, self.sigma_yaw**2)

    def estimate(self, stamp_ns: int) -> PoseEstimate:
        if self._drop_until_ns is not None and stamp_ns >= self._drop_until_ns:
            self._drop_until_ns = None
        x, y, yaw, v = self._sample()
        if self._drop_until_ns is None or self._last is None:
            self._last_fix_ns = stamp_ns
            est = PoseEstimate(stamp_ns, x, y, yaw, v, self._cov.copy(), self.sigma_v**2,
                               abs_fix_ns=self._last_fix_ns)
            self._dr_t0_ns, self._dr_xy = stamp_ns, (x, y)
        else:
            dt = max(0.0, (stamp_ns - self._last.stamp_ns) * 1e-9)
            px, py = self._dr_xy
            px += self._last.v * math.cos(self._last.theta) * dt
            py += self._last.v * math.sin(self._last.theta) * dt
            self._dr_xy = (px, py)
            cov = self._dr_cov((stamp_ns - self._dr_t0_ns) * 1e-9)
            est = PoseEstimate(stamp_ns, px, py, yaw, v, cov, self.sigma_v**2,
                               abs_fix_ns=self._last_fix_ns)
        self._last = est
        return est


class Localization:
    """Always-on front end: one backend in, latest /localization/pose out."""

    def __init__(self, backend: LocBackend, health: HealthMonitor | None = None) -> None:
        self.backend = backend
        self.health = health or HealthMonitor()
        self.pose: PoseEstimate | None = None
        self.healthy = True
        self.health_m = 0.0  # sqrt(P_xx + P_yy) of the latest pose, m
        self.hold_required = False  # HealthMonitor escalation: request supervisor HOLD

    def step(self, stamp_ns: int) -> PoseEstimate:
        self.pose = self.backend.estimate(stamp_ns)
        self.health_m = self.pose.sigma_pos
        self.healthy = self.health.update(self.pose)
        self.hold_required = self.health.hold_required
        return self.pose
