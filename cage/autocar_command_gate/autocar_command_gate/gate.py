# cage/autocar_command_gate/autocar_command_gate/gate.py
"""RSS + occupancy hold + rate/jerk clamp. Frozen 50 Hz. Miss → HOLD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LEAD_CLASSES = {1, 2, 3, 4}  # CAR TRUCK PED CYCLIST


@dataclass(frozen=True, slots=True)
class GateParams:
    a_min_brake_rear: float = 6.0
    a_max_brake_front: float = 8.0
    a_max_accel: float = 2.5
    a_hold: float = 3.0
    rho_s: float = 0.40
    steer_rate_max: float = 0.50
    jerk_max: float = 5.0
    inflate_m: float = 0.05
    horizon_s: float = 1.5
    dt: float = 0.02
    stale_s: float = 0.080
    v_degrade: float = 4.0
    track_m: float = 1.55
    r_inflate: float = 0.30


@dataclass(slots=True)
class Cmd:
    stamp_s: float
    steer: float
    accel: float
    hold: bool
    seq: int
    source: str


@dataclass(slots=True)
class Lead:
    dx: float
    v_rel: float
    v_ego: float
    v_lead: float


def rss_dmin(v_r: float, v_f: float, p: GateParams) -> float:
    """RSS Rule 1: rear uses a_min_brake_rear, front uses a_max_brake_front."""
    a_min_r = p.a_min_brake_rear
    a_max_f = p.a_max_brake_front
    a_max_a = p.a_max_accel
    rho = p.rho_s
    term_r = v_r * rho + 0.5 * a_max_a * rho * rho
    term_r += (v_r + rho * a_max_a) ** 2 / (2.0 * a_min_r)
    term_f = (v_f * v_f) / (2.0 * a_max_f)
    return max(0.0, term_r - term_f)


class _SliceView:
    __slots__ = (
        "width",
        "height",
        "resolution",
        "origin_x",
        "origin_y",
        "cells",
    )

    def __init__(self, width, height, resolution, origin_x, origin_y, cells) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cells = cells


def slice_to_xy(slice_msg: object) -> NDArray[np.float32]:
    """OccupancySlice raster → (N,2) map-frame occupied centroids."""
    w, h = int(slice_msg.width), int(slice_msg.height)
    res = float(slice_msg.resolution)
    ox, oy = float(slice_msg.origin_x), float(slice_msg.origin_y)
    cells = np.asarray(slice_msg.cells, dtype=np.uint8).reshape(h, w)
    jj, ii = np.nonzero(cells == 255)
    if jj.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    xs = ox + (ii.astype(np.float32) + 0.5) * res
    ys = oy + (jj.astype(np.float32) + 0.5) * res
    return np.stack([xs, ys], axis=1)


def occupancy_hit(
    occ_xy: NDArray[np.float32],
    pose_xyyaw: tuple[float, float, float],
    v: float,
    p: GateParams,
) -> bool:
    """True if any occupied point intersects the constant-decel envelope (epsilon only)."""
    if occ_xy.size == 0 or v <= 0.05:
        return False
    x, y, yaw = pose_xyyaw
    c, s = np.cos(yaw), np.sin(yaw)
    n = max(2, int(p.horizon_s / 0.1))
    ts = np.linspace(0.0, p.horizon_s, n, dtype=np.float32)
    decel = p.a_min_brake_rear
    stop_t = v / decel
    dist = np.where(ts < stop_t, v * ts - 0.5 * decel * ts * ts, v * v / (2.0 * decel))
    xs = x + dist * c
    ys = y + dist * s
    d2 = (occ_xy[:, 0:1] - xs[None, :]) ** 2 + (occ_xy[:, 1:2] - ys[None, :]) ** 2
    return bool(np.any(d2 < (p.inflate_m * p.inflate_m)))


def select_lead(
    objects: list[tuple[int, float, float]],
    pose_xyyaw: tuple[float, float, float],
    v_ego: float,
    p: GateParams,
    v_leads: list[float] | None = None,
) -> Lead | None:
    """Closest CLASS_CAR/TRUCK/PED/CYCLIST in the 1.5 s corridor."""
    x, y, yaw = pose_xyyaw
    c, s = np.cos(yaw), np.sin(yaw)
    half = 0.5 * p.track_m + p.r_inflate
    horizon = max(v_ego, 0.1) * p.horizon_s
    best: Lead | None = None
    best_dx = 1e9
    for i, (cls, ox, oy) in enumerate(objects):
        if cls not in LEAD_CLASSES:
            continue
        dx = c * (ox - x) + s * (oy - y)
        dy = -s * (ox - x) + c * (oy - y)
        if dx < 0.0 or dx > horizon or abs(dy) > half:
            continue
        if dx < best_dx:
            best_dx = dx
            v_lead = 0.0 if v_leads is None else v_leads[i]
            best = Lead(dx=dx, v_rel=v_lead - v_ego, v_ego=v_ego, v_lead=v_lead)
    return best


@dataclass(slots=True)
class GateOut:
    cmd: Cmd
    verdict: str
    reason: str


class CommandGate:
    def __init__(self, params: GateParams | None = None) -> None:
        self.p = params or GateParams()
        self._last = Cmd(0.0, 0.0, 0.0, True, 0, "gate")

    def tick(
        self,
        now_s: float,
        incoming: Cmd | None,
        supervisor_state: str,
        lead: Lead | None,
        occ_xy: NDArray[np.float32],
        pose_xyyaw: tuple[float, float, float],
        v_ego: float,
        collision: bool,
    ) -> GateOut:
        comfort = Cmd(now_s, self._last.steer, -self.p.a_hold, True, self._last.seq + 1, "gate")
        panic = Cmd(
            now_s,
            self._last.steer,
            -self.p.a_min_brake_rear,
            True,
            self._last.seq + 1,
            "gate",
        )
        if collision or supervisor_state == "ESTOP":
            self._last = panic
            return GateOut(panic, "hold", "collision" if collision else "supervisor")
        if supervisor_state not in {"RUNNING", "DEGRADED"}:
            self._last = comfort
            return GateOut(comfort, "hold", "supervisor")
        if incoming is None or (now_s - incoming.stamp_s) > self.p.stale_s:
            self._last = comfort
            return GateOut(comfort, "hold", "stale")
        if incoming.hold:
            self._last = comfort
            return GateOut(comfort, "hold", "brain_hold")

        cmd = incoming
        desired_accel = cmd.accel
        if supervisor_state == "DEGRADED":
            if v_ego > self.p.v_degrade + 0.1:
                desired_accel = -self.p.a_hold
                cmd = Cmd(cmd.stamp_s, self._last.steer, desired_accel, False, cmd.seq, cmd.source)
            else:
                desired_accel = 0.0
                cmd = Cmd(cmd.stamp_s, self._last.steer, 0.0, False, cmd.seq, cmd.source)

        if lead is not None and lead.dx < rss_dmin(lead.v_ego, lead.v_lead, self.p):
            self._last = panic
            return GateOut(panic, "hold", "rss")
        if occupancy_hit(occ_xy, pose_xyyaw, v_ego, self.p):
            self._last = panic
            return GateOut(panic, "hold", "occupancy")

        dt = max(1e-3, self.p.dt)
        d_steer = np.clip(
            cmd.steer - self._last.steer,
            -self.p.steer_rate_max * dt,
            self.p.steer_rate_max * dt,
        )
        d_acc = np.clip(
            cmd.accel - self._last.accel,
            -self.p.jerk_max * dt,
            self.p.jerk_max * dt,
        )
        clipped = abs(d_steer - (cmd.steer - self._last.steer)) > 1e-6 or abs(
            d_acc - (cmd.accel - self._last.accel)
        ) > 1e-6
        out = Cmd(
            now_s,
            float(self._last.steer + d_steer),
            float(
                np.clip(
                    self._last.accel + d_acc,
                    -self.p.a_min_brake_rear,
                    self.p.a_max_accel,
                )
            ),
            False,
            cmd.seq,
            cmd.source,
        )
        if lead is not None and lead.dx < rss_dmin(
            max(0.0, lead.v_ego + out.accel * dt), lead.v_lead, self.p
        ):
            self._last = panic
            return GateOut(panic, "hold", "rss")
        self._last = out
        reason = "degraded" if supervisor_state == "DEGRADED" else ("clip" if clipped else "ok")
        return GateOut(out, "clip" if clipped else "pass", reason)
