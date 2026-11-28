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
GNSS innovations via the theta coupling. An optional NDT pose update is
described below; without it the filter is unchanged.

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

GNSS innovation gate. Each fix is tested before the update with the
normalized innovation squared on the 2D position innovation y:

    NIS = y^T S^-1 y,  S = H P H^T + R

and rejected when NIS > CHI2_GATE_2DOF = 11.83, the 99.7 % point of the
chi-square distribution with 2 degrees of freedom. A rejected fix does not
touch the state or the covariance and does not refresh abs_fix_ns, so a run
of rejected fixes ages the absolute fix the same way an outage does. The GNSS
schedule (one fix per 1 / GNSS_RATE_HZ) runs on received fixes, accepted or
not. n_fix_accepted and n_fix_rejected count fixes that reached the gate
(the initializing fix is not counted), and last_nis is the latest NIS.

/scenario/inject/gnss_jump (GNSS_JUMP_SRV) adds a fixed offset of offset_m
(default 5.0 m) to every GNSS fix for estimates in [now, now + duration).
The offset is lateral: along the left normal of the filter heading at the
moment of injection, held fixed in map for the whole jump. It changes no
random draw.

GNSS-suspect latch. The gate alone does not stop a spoof that never ends:
with no accepted fix the dead-reckoning covariance grows until the offset
fix passes the gate (measured at 37 to 47 s for a 5 m offset on the G2
loop). So when the gate has rejected every gated fix for longer than
T_SUSPECT_S (from the first rejected fix of the run to the current one),
gnss_suspect latches. While latched, no GNSS fix is accepted whatever its
NIS (it is counted as rejected), so a growing covariance cannot readmit the
spoof. Any accepted fix ends a run of rejections. A loc_drop produces no
fixes at all, so it never starts a run and never trips the latch. For now
the latch clears only on reset() (the loc side of /reset). NDT does not
clear it either; whether an independent fix may clear it is an open
question in docs/design.md.

NDT pose update (PR-06b). Optional: EkfLoc(..., ndt=Ndt2D, scan=callable)
where scan() returns the current 2D lidar returns as (N, 2) points in the
vehicle frame (for example SimLidar2D.points(BicycleBackend.lidar_scan())).
Without ndt and scan nothing below runs and no random draw changes. Every
1 / NDT_RATE_HZ, after propagation, the matcher runs with the EKF prior at
that stamp as its initial guess. The result is used only when
- the match converged (otherwise n_ndt_not_converged counts it),
- the match took at most ndt_budget_ms (NDT_BUDGET_MS = 30 ms, the design
  budget; otherwise n_ndt_slow counts it and the update is skipped), and
- the 3-DOF NIS on (x, y, yaw) with the yaw innovation wrapped is at most
  CHI2_GATE_3DOF = 14.16, the 99.7 % point of chi-square with 3 degrees of
  freedom (otherwise n_ndt_rejected counts it).
R is the matcher's 3x3 covariance: by default (ndt_r="fixed") the
per-cell information covariance plus the fixed model-error term, or with
ndt_r="hessian" the Laplace covariance from the score Hessian
(NdtResult.cov_hessian), or with ndt_r="hessian_deg" that covariance with
degenerate position directions set to the ceiling
(NdtResult.cov_hessian_deg). The convergence decision is the same for all. An accepted NDT update refreshes
abs_fix_ns like an accepted GNSS fix; a skipped, unconverged or rejected
one does not. loc_drop mutes NDT as well as GNSS (no scan is matched while
it is active). /scenario/inject/gnss_outage (inject_gnss_outage) mutes GNSS
only, so NDT alone can keep the absolute fix fresh.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

import time
from typing import Callable

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
# 99.7 % point of chi-square with 2 dof: P(NIS > 11.83) = 0.0027 for a
# consistent filter. Fixes with a larger NIS are rejected.
CHI2_GATE_2DOF = 11.83
GNSS_JUMP_SRV = "/scenario/inject/gnss_jump"
GNSS_JUMP_M = 5.0
# Rejection run length that latches gnss_suspect. 3.0 s is 1.5 times the 2 s
# catalog fault (a 2 s jump rejects for 1.9 s from its first to its last fix,
# so it never trips), 30 consecutive 10 Hz rejections, which a consistent
# filter at the 0.27 % gate tail practically never produces, and an order of
# magnitude under the 37 s at which the gate alone readmitted a 5 m spoof.
T_SUSPECT_S = 3.0
NDT_RATE_HZ = 10.0
NDT_BUDGET_MS = 30.0
# 99.7 % point of chi-square with 3 dof, same tail as the GNSS gate.
CHI2_GATE_3DOF = 14.16

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

    def __init__(self, gt: GtPose, seed: int | None = None, ndt=None,
                 scan: Callable[[], NDArray[np.float64]] | None = None,
                 ndt_budget_ms: float = NDT_BUDGET_MS, ndt_r: str = "fixed") -> None:
        if (ndt is None) != (scan is None):
            raise ValueError("ndt and scan go together")
        if ndt_r not in ("fixed", "hessian", "hessian_deg"):
            raise ValueError("ndt_r must be 'fixed', 'hessian' or 'hessian_deg'")
        self.ndt_r = ndt_r
        self.gt = gt
        self.ndt, self.scan, self.ndt_budget_ms = ndt, scan, ndt_budget_ms
        self._last_ndt_ns: int | None = None
        self._gnss_out_until_ns: int | None = None
        self.n_ndt_accepted = 0
        self.n_ndt_rejected = 0
        self.n_ndt_not_converged = 0
        self.n_ndt_slow = 0
        self.last_ndt_nis: float | None = None
        self.last_ndt = None  # latest NdtResult
        self._rng = np.random.default_rng(seed)
        self.s = np.zeros(NX)
        self.P = np.zeros((NX, NX))
        self._prev_gt: tuple[float, float, float, float] | None = None
        self._prev_ns: int | None = None
        self._last_fix_ns: int | None = None
        self._drop_until_ns: int | None = None
        self._last_gnss_ns: int | None = None  # last fix received, accepted or not
        self._jump_until_ns: int | None = None
        self._jump_xy = np.zeros(2)
        self.n_fix_accepted = 0
        self.n_fix_rejected = 0
        self.last_nis: float | None = None
        self._reject_run_ns: int | None = None  # first rejected fix of the current run
        self.gnss_suspect = False
        self.t_suspect_ns: int | None = None  # stamp at which the latch tripped

    def reset(self) -> dict:
        """Loc side of /reset: clear the GNSS-suspect latch and the rejection run.
        State and covariance are kept; the next fix is gated as usual."""
        self.gnss_suspect = False
        self.t_suspect_ns = None
        self._reject_run_ns = None
        return {"ok": True, "message": "gnss suspect latch cleared"}

    @property
    def gnss_jump_active(self) -> bool:
        return self._jump_until_ns is not None

    def inject_gnss_jump(self, duration_s: float, now_ns: int, offset_m: float = GNSS_JUMP_M) -> dict:
        """GNSS_JUMP_SRV handler: every fix in [now, now + duration) is offset by
        offset_m to the left of the current filter heading (fixed in map)."""
        if not (math.isfinite(duration_s) and duration_s > 0.0):
            return {"ok": False, "message": "duration_s must be > 0"}
        if not math.isfinite(offset_m):
            return {"ok": False, "message": "offset_m must be finite"}
        if self._jump_until_ns is not None:
            return {"ok": False, "message": "gnss_jump already active"}
        th = float(self.s[STH])
        self._jump_xy = offset_m * np.array([-math.sin(th), math.cos(th)])
        self._jump_until_ns = now_ns + int(round(duration_s * 1e9))
        return {"ok": True, "message": f"gnss offset {offset_m:.2f} m for {duration_s:.3f} s"}

    @property
    def gnss_ok(self) -> bool:
        return self._drop_until_ns is None and self._gnss_out_until_ns is None

    @property
    def ndt_ok(self) -> bool:
        return self.ndt is not None and self._drop_until_ns is None

    def inject_gnss_outage(self, duration_s: float, now_ns: int) -> dict:
        """GNSS only muted for estimates in [now, now + duration). NDT keeps running."""
        if not (math.isfinite(duration_s) and duration_s > 0.0):
            return {"ok": False, "message": "duration_s must be > 0"}
        if self._gnss_out_until_ns is not None:
            return {"ok": False, "message": "gnss_outage already active"}
        self._gnss_out_until_ns = now_ns + int(round(duration_s * 1e9))
        return {"ok": True, "message": f"gnss only muted for {duration_s:.3f} s"}

    def ndt_nis(self, z: NDArray[np.float64], r: NDArray[np.float64]) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """3-DOF NIS of a pose measurement z = (x, y, yaw), yaw innovation wrapped."""
        h = np.zeros((3, NX))
        h[0, SX] = h[1, SY] = h[2, STH] = 1.0
        y = np.array([z[0] - self.s[SX], z[1] - self.s[SY], _wrap(z[2] - self.s[STH])])
        sm = h @ self.P @ h.T + r
        return float(y @ np.linalg.solve(sm, y)), h, y

    def _ndt_update(self) -> bool:
        """Match the current scan from the EKF prior; gated pose update. True if accepted."""
        pts = self.scan()
        t0 = time.perf_counter()
        res = self.ndt.align(pts, (float(self.s[SX]), float(self.s[SY]), float(self.s[STH])))
        ms = (time.perf_counter() - t0) * 1e3
        self.last_ndt = res
        if ms > self.ndt_budget_ms:
            self.n_ndt_slow += 1
            return False
        if not res.converged:
            self.n_ndt_not_converged += 1
            return False
        r = np.asarray({"fixed": res.cov, "hessian": res.cov_hessian,
                        "hessian_deg": res.cov_hessian_deg}[self.ndt_r], float)
        nis, h, y = self.ndt_nis(np.array([res.x, res.y, res.yaw]), r)
        self.last_ndt_nis = nis
        if nis > CHI2_GATE_3DOF:
            self.n_ndt_rejected += 1
            return False
        sm = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(sm)
        self.s = self.s + k @ y
        self.s[STH] = _wrap(self.s[STH])
        ikh = np.eye(NX) - k @ h
        self.P = ikh @ self.P @ ikh.T + k @ r @ k.T
        self.n_ndt_accepted += 1
        return True

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

    def _gnss_update(self, fix: NDArray[np.float64], stamp_ns: int) -> bool:
        """Gated GNSS position update. Returns True when the fix is accepted."""
        h = np.zeros((2, NX))
        h[0, SX] = h[1, SY] = 1.0
        r = SIGMA_GNSS_M**2 * np.eye(2)
        y = fix - h @ self.s
        sm = h @ self.P @ h.T + r
        nis = float(y @ np.linalg.solve(sm, y))
        self.last_nis = nis
        if self.gnss_suspect or nis > CHI2_GATE_2DOF:
            self.n_fix_rejected += 1
            if self._reject_run_ns is None:
                self._reject_run_ns = stamp_ns
            if not self.gnss_suspect and stamp_ns - self._reject_run_ns > int(round(T_SUSPECT_S * 1e9)):
                self.gnss_suspect = True
                self.t_suspect_ns = stamp_ns
            return False
        self._reject_run_ns = None
        self._update(h, fix, r)
        self.n_fix_accepted += 1
        return True

    def estimate(self, stamp_ns: int) -> PoseEstimate:
        if self._drop_until_ns is not None and stamp_ns >= self._drop_until_ns:
            self._drop_until_ns = None
        if self._jump_until_ns is not None and stamp_ns >= self._jump_until_ns:
            self._jump_until_ns = None
        if self._gnss_out_until_ns is not None and stamp_ns >= self._gnss_out_until_ns:
            self._gnss_out_until_ns = None
        gx, gy, gyaw, gv = (float(c) for c in self.gt())
        n = self._rng.standard_normal(6)  # gyro, accel, odom v, gnss x, gnss y, heading0
        fix = np.array([gx + SIGMA_GNSS_M * n[3], gy + SIGMA_GNSS_M * n[4]])
        if self._jump_until_ns is not None:
            fix = fix + self._jump_xy
        odom_v = gv + SIGMA_ODOM_V * n[2]
        if self._prev_gt is None:
            self.s = np.array([fix[0], fix[1], _wrap(gyaw + SIGMA_THETA0 * n[5]), odom_v, 0.0])
            self.P = np.diag([SIGMA_GNSS_M**2, SIGMA_GNSS_M**2, SIGMA_THETA0**2, SIGMA_ODOM_V**2, SIGMA_BIAS0**2])
            self._last_fix_ns = self._last_gnss_ns = stamp_ns
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
            due = self._last_gnss_ns is None or (stamp_ns - self._last_gnss_ns) * 1e-9 >= 1.0 / GNSS_RATE_HZ - 1e-6
            if self.gnss_ok and due:
                self._last_gnss_ns = stamp_ns
                if self._gnss_update(fix, stamp_ns):
                    self._last_fix_ns = stamp_ns
            if self.ndt is not None:
                ndt_due = (self._last_ndt_ns is None
                           or (stamp_ns - self._last_ndt_ns) * 1e-9 >= 1.0 / NDT_RATE_HZ - 1e-6)
                if self.ndt_ok and ndt_due:
                    self._last_ndt_ns = stamp_ns
                    if self._ndt_update():
                        self._last_fix_ns = stamp_ns
        self._prev_gt, self._prev_ns = (gx, gy, gyaw, gv), stamp_ns
        return PoseEstimate(stamp_ns, float(self.s[SX]), float(self.s[SY]), float(self.s[STH]),
                            float(self.s[SV]), self.pose_cov(), float(self.P[SV, SV]),
                            abs_fix_ns=self._last_fix_ns, abs_suspect=self.gnss_suspect)

    def pose_cov(self) -> NDArray[np.float64]:
        """6x6 pose covariance from the filter's x, y, theta block."""
        p = self.P
        c = pose_covariance(p[SX, SX], p[SY, SY], p[STH, STH], p[SX, SY])
        c[IX, IYAW] = c[IYAW, IX] = p[SX, STH]
        c[IY, IYAW] = c[IYAW, IY] = p[SY, STH]
        return c
