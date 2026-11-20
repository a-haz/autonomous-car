"""EKF localization backend (PR-06b slice 2). Pure Python, no ROS.

State (docs/design.md, Localization): s = [x, y, theta, v, b_az] in map.

Propagate at the estimate rate (50 Hz) with the rear-axle kinematic bicycle,
using the gyro yaw rate minus the bias and the longitudinal accelerometer,
over dt between stamps:

    w      = r - b_az
    th_m   = theta + w dt / 2
    x'     = x + v dt cos(th_m)
    y'     = y + v dt sin(th_m)
    theta' = theta + w dt
    v'     = v + a dt
    b_az'  = b_az            (random walk)

Updates: wheel-odometry speed on v every estimate, and a GNSS position fix on
(x, y) at GNSS_RATE_HZ only while gnss_ok. b_az is observed through the
GNSS innovations via the theta coupling. NDT is not part of this slice.

Sensors are synthetic and seeded. They are derived from the ground truth the
backend is given: yaw rate and accel are the GT differences between two
consecutive estimates, so they are the exact averages over the interval the
plant moved through. Noise values, chosen from typical sensor specs:

    GNSS position     SIGMA_GNSS_M   = 0.30 m per horizontal axis, white, 10 Hz
                      (decimetre-class dual-band receiver, SBAS or RTK float)
    gyro z            SIGMA_GYRO     = 0.005 rad/s white per 50 Hz sample
                      GYRO_BIAS_TRUE = 0.01 rad/s constant true bias
                      (MEMS automotive IMU, turn-on bias under 1 deg/s)
                      Q_BIAS         = (1e-4)^2 rad^2/s^3 bias random walk in the filter
                      SIGMA_BIAS0    = 0.02 rad/s initial bias sigma
    accelerometer x   SIGMA_ACCEL    = 0.05 m/s^2 white per 50 Hz sample
    wheel odom speed  SIGMA_ODOM_V   = 0.05 m/s white per 50 Hz sample
    initial heading   SIGMA_THETA0   = 0.05 rad (alignment error at start)
    model error       Q_POS          = 1e-4 m^2/s per axis on x, y

The first estimate initializes from a GNSS fix, the noisy heading and the
odometry speed. The 6x6 output covariance carries the filter's x, y, yaw
block (with cross terms); z, roll and pitch stay at COV_UNUSED.

/scenario/inject/loc_drop has the same semantics as GtInsStub: no GNSS fix
for estimates in [now, now + duration). Every random draw happens on every
estimate, so a run with and without a drop sees the same noise stream.
abs_fix_ns on each PoseEstimate is the stamp of the last accepted GNSS fix
(the initializing fix counts).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .gt_ins import GtPose, _wrap
from .pose import IX, IY, IYAW, PoseEstimate, pose_covariance

SIGMA_GNSS_M = 0.30
GNSS_RATE_HZ = 10.0
SIGMA_GYRO = 0.005
GYRO_BIAS_TRUE = 0.01
Q_BIAS = 1e-4**2
SIGMA_BIAS0 = 0.02
SIGMA_ACCEL = 0.05
SIGMA_ODOM_V = 0.05
SIGMA_THETA0 = 0.05
Q_POS = 1e-4

NX = 5
SX, SY, STH, SV, SB = range(NX)


def predict(s: NDArray[np.float64], r: float, a: float, dt: float) -> NDArray[np.float64]:
    """Bicycle propagate of s = [x, y, theta, v, b_az] with gyro r and accel a."""
    x, y, th, v, b = s
    w = r - b
    thm = th + 0.5 * w * dt
    return np.array([x + v * dt * math.cos(thm), y + v * dt * math.sin(thm), _wrap(th + w * dt), v + a * dt, b])


def jacobians(s: NDArray[np.float64], r: float, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """F = d predict / d s and G = d predict / d (r, a)."""
    _, _, th, v, b = s
    thm = th + 0.5 * (r - b) * dt
    c, sn = math.cos(thm), math.sin(thm)
    f = np.eye(NX)
    f[SX, STH], f[SX, SV], f[SX, SB] = -v * dt * sn, dt * c, 0.5 * v * dt * dt * sn
    f[SY, STH], f[SY, SV], f[SY, SB] = v * dt * c, dt * sn, -0.5 * v * dt * dt * c
    f[STH, SB] = -dt
    g = np.zeros((NX, 2))
    g[SX, 0], g[SY, 0], g[STH, 0] = -0.5 * v * dt * dt * sn, 0.5 * v * dt * dt * c, dt
    g[SV, 1] = dt
    return f, g


class EkfLoc:
    """LocBackend: EKF over synthetic gyro, accel, wheel odom and GNSS from GT."""

    def __init__(self, gt: GtPose, seed: int | None = None) -> None:
        self.gt = gt
        self._rng = np.random.default_rng(seed)
        self.s = np.zeros(NX)
        self.P = np.zeros((NX, NX))
        self._prev_gt: tuple[float, float, float, float] | None = None
        self._prev_ns: int | None = None
        self._last_fix_ns: int | None = None
        self._drop_until_ns: int | None = None

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

    def _update(self, h: NDArray[np.float64], z: NDArray[np.float64], r: NDArray[np.float64]) -> None:
        y = z - h @ self.s
        sm = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(sm)
        self.s = self.s + k @ y
        self.s[STH] = _wrap(self.s[STH])
        ikh = np.eye(NX) - k @ h
        self.P = ikh @ self.P @ ikh.T + k @ r @ k.T  # Joseph form

    def estimate(self, stamp_ns: int) -> PoseEstimate:
        if self._drop_until_ns is not None and stamp_ns >= self._drop_until_ns:
            self._drop_until_ns = None
        gx, gy, gyaw, gv = (float(c) for c in self.gt())
        n = self._rng.standard_normal(6)  # gyro, accel, odom v, gnss x, gnss y, heading0
        fix = np.array([gx + SIGMA_GNSS_M * n[3], gy + SIGMA_GNSS_M * n[4]])
        odom_v = gv + SIGMA_ODOM_V * n[2]
        if self._prev_gt is None:
            self.s = np.array([fix[0], fix[1], _wrap(gyaw + SIGMA_THETA0 * n[5]), odom_v, 0.0])
            self.P = np.diag([SIGMA_GNSS_M**2, SIGMA_GNSS_M**2, SIGMA_THETA0**2, SIGMA_ODOM_V**2, SIGMA_BIAS0**2])
            self._last_fix_ns = stamp_ns
        else:
            dt = (stamp_ns - self._prev_ns) * 1e-9
            if dt > 0.0:
                px, py, pyaw, pv = self._prev_gt
                r = _wrap(gyaw - pyaw) / dt + GYRO_BIAS_TRUE + SIGMA_GYRO * n[0]
                a = (gv - pv) / dt + SIGMA_ACCEL * n[1]
                f, g = jacobians(self.s, r, dt)
                self.s = predict(self.s, r, a, dt)
                q = g @ np.diag([SIGMA_GYRO**2, SIGMA_ACCEL**2]) @ g.T
                q[SX, SX] += Q_POS * dt
                q[SY, SY] += Q_POS * dt
                q[SB, SB] += Q_BIAS * dt
                self.P = f @ self.P @ f.T + q
            h_v = np.zeros((1, NX))
            h_v[0, SV] = 1.0
            self._update(h_v, np.array([odom_v]), np.array([[SIGMA_ODOM_V**2]]))
            due = self._last_fix_ns is None or (stamp_ns - self._last_fix_ns) * 1e-9 >= 1.0 / GNSS_RATE_HZ - 1e-6
            if self.gnss_ok and due:
                h_p = np.zeros((2, NX))
                h_p[0, SX] = h_p[1, SY] = 1.0
                self._update(h_p, fix, SIGMA_GNSS_M**2 * np.eye(2))
                self._last_fix_ns = stamp_ns
        self._prev_gt, self._prev_ns = (gx, gy, gyaw, gv), stamp_ns
        return PoseEstimate(stamp_ns, float(self.s[SX]), float(self.s[SY]), float(self.s[STH]),
                            float(self.s[SV]), self.pose_cov(), float(self.P[SV, SV]),
                            abs_fix_ns=self._last_fix_ns)

    def pose_cov(self) -> NDArray[np.float64]:
        """6x6 pose covariance from the filter's x, y, theta block."""
        p = self.P
        c = pose_covariance(p[SX, SX], p[SY, SY], p[STH, STH], p[SX, SY])
        c[IX, IYAW] = c[IYAW, IX] = p[SX, STH]
        c[IY, IYAW] = c[IYAW, IY] = p[SY, STH]
        return c
