"""G1 closed loop: GT pose -> centerline planner -> shared StanleyPID -> bicycle.

GT objects go through autocar_perc_gt, so the planner only sees
TrackedObjects. No ROS, no Isaac, no gate in this loop.
Bar from docs/design.md Verification Gates: 3 seeds, 0 collision, lat err < 0.50 m.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from autocar_ctrl.bicycle import L_TWIN, BicycleParams, BicycleState
from autocar_ctrl.stanley import StanleyPID
from autocar_perc_gt.stub import CLASS_STATIC, GtObj, to_tracked
from autocar_plan.simple_planner import HALF_W, TRACK_M, CenterlinePlanner, closest_frenet
from autocar_vehicle_interface.bicycle import BicycleBackend

import make_figure8

ROOT = Path(__file__).resolve().parents[4]
CENTERLINE = ROOT / "maps" / "figure8_v0" / "centerline.csv"
DT_CTRL = 0.02  # 50 Hz control
PLAN_EVERY = 5  # 10 Hz planner
V_CRUISE = 5.0
LAT_ERR_BAR_M = 0.50
LANELET_ID = 1
# (start fraction of the lap, lateral start offset in m, + is left)
SEEDS = ((0.0, 0.0), (1.0 / 3.0, 0.3), (2.0 / 3.0, -0.3))


def _lane() -> np.ndarray:
    pts = make_figure8.load_centerline(CENTERLINE)
    xy = np.array([(p[0], p[1]) for p in pts], dtype=np.float64)
    return xy[:-1] if np.hypot(*(xy[-1] - xy[0])) < 1e-9 else xy


def _footprint(x: float, y: float, yaw: float, rear: float, front: float, half_w: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    loc = ((-rear, -half_w), (front, -half_w), (front, half_w), (-rear, half_w))
    return np.array([(x + c * a - s * b, y + s * a + c * b) for a, b in loc])


def _car_poly(st: BicycleState) -> np.ndarray:
    """base_link (rear axle) rectangle L x track, per the design's collision check."""
    return _footprint(st.x, st.y, st.yaw, 0.0, L_TWIN, 0.5 * TRACK_M)


def _obj_poly(o: GtObj) -> np.ndarray:
    return _footprint(o.x, o.y, o.yaw, 0.5 * o.l, 0.5 * o.l, 0.5 * o.w)


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(4):
            ex, ey = poly[(i + 1) % 4] - poly[i]
            n = np.array([-ey, ex])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    u = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab)))


def _poly_dist(a: np.ndarray, b: np.ndarray) -> float:
    if _overlap(a, b):
        return 0.0
    d = math.inf
    for p, q in ((a, b), (b, a)):
        for v in p:
            for i in range(4):
                d = min(d, _seg_dist(v, q[i], q[(i + 1) % 4]))
    return d


class _LatErr:
    """Distance of a point to the lane, with a station window so the
    figure-8 crossing cannot snap onto the other branch."""

    def __init__(self, lane: np.ndarray, idx: int) -> None:
        self.lane, self.idx, self.m = lane, idx, len(lane)

    def __call__(self, x: float, y: float) -> float:
        cand = (self.idx + np.arange(-40, 41)) % self.m
        d = np.hypot(self.lane[cand, 0] - x, self.lane[cand, 1] - y)
        self.idx = int(cand[int(np.argmin(d))])
        p = np.array([x, y])
        return min(_seg_dist(p, self.lane[(self.idx + k) % self.m], self.lane[(self.idx + k + 1) % self.m])
                   for k in (-1, 0))


def _start(lane: np.ndarray, frac: float, off: float) -> tuple[int, BicycleState]:
    i = int(round(frac * len(lane))) % len(lane)
    a, b = lane[i], lane[(i + 1) % len(lane)]
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    return i, BicycleState(a[0] - off * math.sin(yaw), a[1] + off * math.cos(yaw), yaw, 0.0)


def _run(frac: float, off: float, objects: list[GtObj], laps: float = 1.0, t_max: float = 200.0) -> dict:
    lane = _lane()
    i0, st0 = _start(lane, frac, off)
    backend = BicycleBackend(st0, BicycleParams())
    ctrl = StanleyPID(L_TWIN)
    planner = CenterlinePlanner(lane, LANELET_ID, V_CRUISE)
    lat_rear, lat_front = _LatErr(lane, i0), _LatErr(lane, i0)
    ds = float(np.mean(np.linalg.norm(np.diff(lane, axis=0), axis=1)))
    lap_m = ds * len(lane)
    obj_polys = [_obj_poly(o) for o in objects]
    rear_err: list[float] = []
    front_err: list[float] = []
    collisions = 0
    min_clear = math.inf
    traveled = 0.0
    traj = None
    t_stop_cmd = None
    t = 0.0
    lights = {"lights": []}
    for k in range(int(t_max / DT_CTRL)):
        st = backend.state
        if k % PLAN_EVERY == 0:
            tracked = to_tracked(int(t * 1e9), objects)
            traj = planner.step((st.x, st.y, st.yaw), tracked, lights)
            if t_stop_cmd is None and traj[0, 4] == 0.0:
                t_stop_cmd = t
        fx, fy = st.x + L_TWIN * math.cos(st.yaw), st.y + L_TWIN * math.sin(st.yaw)
        e, th, vref, _ = closest_frenet((fx, fy, st.yaw), traj)
        backend.set_command(ctrl.lateral(e, th, st.v), ctrl.longitudinal(st.v, vref, DT_CTRL))
        prev = (st.x, st.y)
        st = backend.step(DT_CTRL)
        t += DT_CTRL
        traveled += math.hypot(st.x - prev[0], st.y - prev[1])
        rear_err.append(lat_rear(st.x, st.y))
        front_err.append(lat_front(st.x + L_TWIN * math.cos(st.yaw), st.y + L_TWIN * math.sin(st.yaw)))
        car = _car_poly(st)
        for op in obj_polys:
            if _overlap(car, op):
                collisions += 1
            min_clear = min(min_clear, _poly_dist(car, op))
        if traveled >= laps * lap_m:
            break
        if t_stop_cmd is not None and st.v <= 1e-3:
            break
    rr, ff = np.array(rear_err), np.array(front_err)
    return {
        "max_rear": float(rr.max()), "rms_rear": float(np.sqrt(np.mean(rr**2))),
        "max_front": float(ff.max()), "rms_front": float(np.sqrt(np.mean(ff**2))),
        "collisions": collisions, "min_clear": min_clear, "traveled": traveled, "lap_m": lap_m,
        "t": t, "v": backend.state.v, "t_stop_cmd": t_stop_cmd, "state": backend.state,
    }


def _roadside_objects(lane: np.ndarray) -> list[GtObj]:
    """Static boxes beside the lane, outside the corridor everywhere on the lap."""
    out = []
    for n, frac in enumerate((0.15, 0.4, 0.65, 0.9)):
        i = int(frac * len(lane))
        a, b = lane[i], lane[i + 1]
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        side = 3.5
        o = GtObj(100 + n, CLASS_STATIC, a[0] - side * math.sin(yaw), a[1] + side * math.cos(yaw), yaw, 1.0, 1.0, 1.0)
        assert np.min(np.hypot(lane[:, 0] - o.x, lane[:, 1] - o.y)) > HALF_W + 1.0
        out.append(o)
    return out


@pytest.mark.parametrize("frac,off", SEEDS)
def test_g1_closed_loop_figure8(frac: float, off: float) -> None:
    lane = _lane()
    r = _run(frac, off, _roadside_objects(lane))
    print(f"G1 seed start={frac:.3f} off={off:+.2f}  max_rear={r['max_rear']:.4f} rms_rear={r['rms_rear']:.4f}  "
          f"max_front={r['max_front']:.4f} rms_front={r['rms_front']:.4f}  collisions={r['collisions']} "
          f"min_clear={r['min_clear']:.3f} lap={r['traveled']:.1f}/{r['lap_m']:.1f} m t={r['t']:.1f} s")
    assert r["traveled"] >= r["lap_m"]
    assert r["t_stop_cmd"] is None
    assert r["collisions"] == 0
    assert r["max_rear"] < LAT_ERR_BAR_M
    assert r["max_front"] < LAT_ERR_BAR_M


def test_g1_static_obstacle_in_corridor_stops_before_contact() -> None:
    lane = _lane()
    i = int(round(40.0 / 0.25))
    a, b = lane[i], lane[i + 1]
    obj = GtObj(200, CLASS_STATIC, a[0], a[1], math.atan2(b[1] - a[1], b[0] - a[0]), 1.0, 1.0, 1.0)
    r = _run(0.0, 0.0, [obj], t_max=60.0)
    st = r["state"]
    front = np.array([st.x + L_TWIN * math.cos(st.yaw), st.y + L_TWIN * math.sin(st.yaw)])
    gap = float(np.linalg.norm(front - np.array([obj.x, obj.y]))) - 0.5 * obj.l
    print(f"G1 obstacle  stop_cmd_t={r['t_stop_cmd']:.2f} s  stopped_t={r['t']:.2f} s  v={r['v']:.4f}  "
          f"front_axle_to_obj_face={gap:.3f} m  min_clear={r['min_clear']:.3f} m  collisions={r['collisions']}  "
          f"max_rear={r['max_rear']:.4f}")
    assert r["t_stop_cmd"] is not None
    assert r["v"] <= 1e-3
    assert r["collisions"] == 0
    assert r["min_clear"] > 0.0
