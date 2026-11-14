"""G1 centerline follower: NOT a lattice. Stop if red or STATIC in corridor.

Moving actors are the gate's RSS job (cut_in must not zero v here).
Pure Python core; the ROS node publishes plan() output on
/brains/a/trajectory and the shared stanley_pid_node turns it into
/brains/a/command. This package has no Stanley of its own.

Trajectory rows are (t, x, y, yaw, v), 64 x 0.1 s (docs/design.md).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from autocar_ctrl.bicycle import wrap

N = 64
DT = 0.1
# Twin values from docs/design.md (command gate table): track 1.55 m, r_inflate 0.30 m.
TRACK_M = 1.55
R_INFLATE = 0.30
HALF_W = 0.5 * TRACK_M + R_INFLATE
# Objects slower than this count as static for the G1 stop rule.
V_STATIC = 0.2

CLASS_TRAFFIC_LIGHT = 5
CLASS_STATIC = 6
LIGHT_RED = 1


def _arc(xy: NDArray[np.float64]) -> NDArray[np.float64]:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(d)))


def resample_centerline(xy: NDArray[np.float64], v_ref: float) -> NDArray[np.float64]:
    """xy (M,2) map points ahead of the car to (64, 5) = t,x,y,yaw,v along arc length."""
    if len(xy) < 2:
        raise ValueError("centerline")
    s = _arc(xy)
    total = float(s[-1])
    t = np.arange(N) * DT
    s_q = np.clip(v_ref * t, 0.0, total)
    x = np.interp(s_q, s, xy[:, 0])
    y = np.interp(s_q, s, xy[:, 1])
    # Heading from the source polyline, not from the samples, so it stays
    # defined even when samples coincide.
    seg_yaw = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    idx = np.clip(np.searchsorted(s, s_q, side="right") - 1, 0, len(seg_yaw) - 1)
    yaw = seg_yaw[idx]
    return np.stack([t, x, y, yaw, np.full(N, v_ref)], axis=1)


def static_xy(tracked: dict) -> NDArray[np.float64]:
    """Map-frame centers of static objects from a TrackedObjects-shaped dict."""
    pts = []
    for o in tracked.get("objects", []):
        if o["classification"] == CLASS_TRAFFIC_LIGHT:
            continue
        vx, vy = o["twist"][0], o["twist"][1]
        if o["classification"] == CLASS_STATIC or math.hypot(vx, vy) < V_STATIC:
            pts.append((o["pose"][0], o["pose"][1]))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def static_in_path(traj: NDArray[np.float64], static_pts: NDArray[np.float64], half_w: float) -> bool:
    """half_w = track/2 + r_inflate. Only STATIC objects (not moving leads)."""
    for px, py in static_pts:
        d = np.hypot(traj[:, 1] - px, traj[:, 2] - py)
        if np.any(d < half_w):
            return True
    return False


def light_is_red(lights: dict, lanelet_id: int) -> bool:
    """Bind to the Lanelet element on /map/lights, never the nearest bulb."""
    for lt in lights.get("lights", []):
        if int(lt["lanelet_id"]) == int(lanelet_id):
            return int(lt["light_state"]) == LIGHT_RED
    return False


def plan(centerline_xy: NDArray[np.float64], v_cruise: float, static_pts: NDArray[np.float64],
         light_red: bool, half_w: float = HALF_W) -> NDArray[np.float64]:
    """Resample the lane ahead at v_cruise; zero v if red or a static object is in the corridor.

    The geometry is kept when stopping so the controller still has a heading.
    """
    traj = resample_centerline(centerline_xy, v_cruise)
    if light_red or static_in_path(traj, static_pts, half_w):
        traj[:, 4] = 0.0
    return traj


def closest_frenet(pose: tuple[float, float, float], traj: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """(e, theta_e, v_ref, path_yaw) from the closest trajectory segment.

    e is positive when the path is to the LEFT of the pose (StanleyPID sign).
    theta_e is path yaw minus vehicle yaw.
    """
    x, y, yaw = pose
    i = int(np.argmin(np.hypot(traj[:, 1] - x, traj[:, 2] - y)))
    best = None
    for j in (i - 1, i):
        if j < 0 or j + 1 >= len(traj):
            continue
        ax, ay, bx, by = traj[j, 1], traj[j, 2], traj[j + 1, 1], traj[j + 1, 2]
        sx, sy = bx - ax, by - ay
        seg2 = sx * sx + sy * sy
        u = 0.0 if seg2 < 1e-12 else min(1.0, max(0.0, ((x - ax) * sx + (y - ay) * sy) / seg2))
        px, py = ax + u * sx, ay + u * sy
        d = math.hypot(x - px, y - py)
        if best is None or d < best[0]:
            tyaw = traj[j, 3] + u * wrap(traj[j + 1, 3] - traj[j, 3])
            vref = traj[j, 4] + u * (traj[j + 1, 4] - traj[j, 4])
            best = (d, px, py, wrap(tyaw), vref)
    if best is None:
        tx, ty, tyaw, vref = traj[i, 1], traj[i, 2], traj[i, 3], traj[i, 4]
    else:
        _, tx, ty, tyaw, vref = best
    e = -math.sin(tyaw) * (tx - x) + math.cos(tyaw) * (ty - y)
    return e, wrap(tyaw - yaw), float(vref), float(tyaw)


class CenterlinePlanner:
    """Tracks the car's station on a closed lane so the figure-8 crossing
    cannot snap the plan onto the other branch, then calls plan()."""

    def __init__(self, lane_xy: NDArray[np.float64], lanelet_id: int, v_cruise: float,
                 half_w: float = HALF_W, closed: bool = True) -> None:
        lane = np.asarray(lane_xy, dtype=np.float64)
        if closed and np.hypot(*(lane[-1] - lane[0])) < 1e-6:
            lane = lane[:-1]
        self.lane = lane
        self.closed = closed
        self.lanelet_id = lanelet_id
        self.v_cruise = v_cruise
        self.half_w = half_w
        self.idx: int | None = None
        ds = np.linalg.norm(np.diff(lane, axis=0), axis=1)
        self._ds = float(np.mean(ds))

    def _locate(self, x: float, y: float, window_m: float = 10.0) -> int:
        m = len(self.lane)
        if self.idx is None:
            cand = np.arange(m)
        else:
            w = int(math.ceil(window_m / self._ds))
            cand = (self.idx + np.arange(-w, w + 1)) % m if self.closed else np.clip(
                self.idx + np.arange(-w, w + 1), 0, m - 1)
        d = np.hypot(self.lane[cand, 0] - x, self.lane[cand, 1] - y)
        self.idx = int(cand[int(np.argmin(d))])
        return self.idx

    def lane_ahead(self, x: float, y: float) -> NDArray[np.float64]:
        i = self._locate(x, y)
        need = int(math.ceil(self.v_cruise * DT * (N - 1) / self._ds)) + 2
        m = len(self.lane)
        if self.closed:
            idx = (i + np.arange(need)) % m
        else:
            idx = np.arange(i, min(i + need, m))
        return self.lane[idx]

    def step(self, pose: tuple[float, float, float], tracked: dict, lights: dict) -> NDArray[np.float64]:
        ahead = self.lane_ahead(pose[0], pose[1])
        return plan(ahead, self.v_cruise, static_xy(tracked), light_is_red(lights, self.lanelet_id), self.half_w)
