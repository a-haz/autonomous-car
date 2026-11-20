"""Localization interface. Pure Python, no ROS.

Every brain and the shared controller read pose from /localization/pose,
never from the simulator directly. The covariance follows
geometry_msgs/PoseWithCovariance: 6x6 over (x, y, z, rot_x, rot_y, rot_z),
row-major, with the unused axes (z, roll, pitch) set large
(docs/design.md, Localization).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

POSE_TOPIC = "/localization/pose"
ODOM_TOPIC = "/localization/odom"
RATE_HZ = 50.0
FRAME_ID = "map"

# Index of each axis in the 6x6 pose covariance.
IX, IY, IZ, IROLL, IPITCH, IYAW = range(6)
COV_UNUSED = 1e6

# Loc is unhealthy when sqrt(P_xx + P_yy) stays above the bar, or the last
# absolute fix stays older than T_ABS_STALE_S, for > 1.0 s.
UNHEALTHY_SIM_M = 0.8
UNHEALTHY_LOT_M = 0.4
UNHEALTHY_HOLD_S = 1.0
# Absolute-fix freshness: a fix (GNSS now, NDT later) older than this counts as
# bad even when the covariance is small. 0.5 s is five missed 10 Hz GNSS fixes.
T_ABS_STALE_S = 0.5


def pose_covariance(var_x: float, var_y: float, var_yaw: float, cov_xy: float = 0.0) -> NDArray[np.float64]:
    """6x6 pose covariance with x, y, yaw filled and the unused axes large."""
    if var_x < 0.0 or var_y < 0.0 or var_yaw < 0.0:
        raise ValueError("variance must be >= 0")
    p = np.zeros((6, 6))
    p[IX, IX], p[IY, IY], p[IYAW, IYAW] = var_x, var_y, var_yaw
    p[IX, IY] = p[IY, IX] = cov_xy
    for i in (IZ, IROLL, IPITCH):
        p[i, i] = COV_UNUSED
    return p


def position_uncertainty(cov: NDArray[np.float64]) -> float:
    """sqrt(P_xx + P_yy) in m, from a 6x6, 3x3 or 2x2 covariance."""
    c = np.asarray(cov, dtype=np.float64)
    if c.shape == (36,):
        c = c.reshape(6, 6)
    return math.sqrt(float(c[IX, IX] + c[IY, IY]))


@dataclass(slots=True)
class PoseEstimate:
    stamp_ns: int
    x: float
    y: float
    theta: float
    v: float
    cov: NDArray[np.float64] = field(default_factory=lambda: pose_covariance(0.0, 0.0, 0.0))
    var_v: float = 0.0
    frame_id: str = FRAME_ID
    # Stamp of the backend's last accepted absolute update (GNSS now, NDT
    # later). None means the backend does not report it.
    abs_fix_ns: int | None = None

    @property
    def abs_age_s(self) -> float | None:
        """Seconds since the last absolute fix, or None if not reported."""
        if self.abs_fix_ns is None:
            return None
        return (self.stamp_ns - self.abs_fix_ns) * 1e-9

    @property
    def sigma_pos(self) -> float:
        return position_uncertainty(self.cov)

    def xyyaw(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    def to_msg(self) -> dict:
        """PoseWithCovarianceStamped-shaped dict for POSE_TOPIC."""
        h = 0.5 * self.theta
        return {
            "header": {"stamp_ns": self.stamp_ns, "frame_id": self.frame_id},
            "pose": {
                "position": (self.x, self.y, 0.0),
                "orientation": (0.0, 0.0, math.sin(h), math.cos(h)),
            },
            "covariance": [float(v) for v in np.asarray(self.cov).reshape(36)],
        }


class HealthMonitor:
    """Healthy until a pose is bad for longer than hold_s.

    A pose is bad when sqrt(P_xx + P_yy) > bar_m, or when its last absolute
    fix is older than t_abs_stale_s. The freshness check catches a GNSS or
    NDT outage while dead reckoning still keeps the covariance small. A pose
    with no abs_fix_ns is judged on the covariance alone.
    """

    def __init__(self, bar_m: float = UNHEALTHY_SIM_M, hold_s: float = UNHEALTHY_HOLD_S,
                 t_abs_stale_s: float = T_ABS_STALE_S) -> None:
        self.bar_m, self.hold_s, self.t_abs_stale_s = bar_m, hold_s, t_abs_stale_s
        self._over_since: int | None = None

    def bad(self, est: PoseEstimate) -> bool:
        age = est.abs_age_s
        return est.sigma_pos > self.bar_m or (age is not None and age > self.t_abs_stale_s)

    def update(self, est: PoseEstimate) -> bool:
        if not self.bad(est):
            self._over_since = None
            return True
        if self._over_since is None:
            self._over_since = est.stamp_ns
        return (est.stamp_ns - self._over_since) * 1e-9 <= self.hold_s
