"""PR-03a bicycle twin tests. No ROS, no Isaac."""
from __future__ import annotations

import math
from pathlib import Path

import make_figure8
from autocar_ctrl.bicycle import L_TWIN, BicycleState, bicycle_step, wrap
from autocar_ctrl.stanley import StanleyPID
from autocar_vehicle_interface.bicycle import BicycleBackend

ROOT = Path(__file__).resolve().parents[3]
CENTERLINE = ROOT / "maps" / "figure8_v0" / "centerline.csv"
DT = 0.02
V_CRUISE = 5.0
# Catalog G1 bar from docs/design.md. Applied only if the measured max
# rear-axle cross-track on this figure-8 sits honestly under it.
LAT_ERR_BAR_M = 0.50


def test_open_loop_canned_command_matches_exact_arc() -> None:
    v = V_CRUISE
    backend = BicycleBackend(BicycleState(0.0, 0.0, 0.0, v))
    backend.set_command(0.0, 0.0)
    n_straight = 100
    for _ in range(n_straight):
        backend.step(DT)
    t_straight = n_straight * DT
    x0, y0, yaw0 = bicycle_step(0.0, 0.0, 0.0, v, 0.0, t_straight, L_TWIN)
    assert math.isclose(backend.state.x, x0, abs_tol=1e-12)
    assert math.isclose(backend.state.y, y0, abs_tol=1e-12)
    assert math.isclose(backend.state.yaw, yaw0, abs_tol=1e-12)
    assert math.isclose(backend.state.x, v * t_straight, abs_tol=1e-12)

    delta = 0.30
    backend.set_command(delta, 0.0)
    n_arc = 250
    for _ in range(n_arc):
        backend.step(DT)
    t_arc = n_arc * DT
    x1, y1, yaw1 = bicycle_step(x0, y0, yaw0, v, delta, t_arc, L_TWIN)
    assert math.isclose(backend.state.x, x1, abs_tol=1e-9)
    assert math.isclose(backend.state.y, y1, abs_tol=1e-9)
    assert math.isclose(wrap(backend.state.yaw - yaw1), 0.0, abs_tol=1e-12)
    assert math.isclose(backend.state.v, v, abs_tol=1e-12)

    r = L_TWIN / math.tan(delta)
    cx, cy = x0, y0 + r
    assert math.isclose(math.hypot(backend.state.x - cx, backend.state.y - cy), r, abs_tol=1e-9)


def test_figure8_centerline_is_checked_in_and_deterministic() -> None:
    assert CENTERLINE.is_file()
    assert make_figure8.OUT_PATH == CENTERLINE
    committed = CENTERLINE.read_text()
    make_figure8.write_centerline(CENTERLINE)
    assert CENTERLINE.read_text() == committed
    pts = make_figure8.load_centerline(CENTERLINE)
    assert pts[0] == (0.0, 0.0, 0.0, 0.0)
    assert pts[-1] == (0.0, 0.0, 0.0, 0.0)
    assert len(pts) > 100
    origins = [i for i, p in enumerate(pts) if p == (0.0, 0.0, 0.0, 0.0)]
    assert origins[0] == 0 and origins[-1] == len(pts) - 1
    assert any(0 < i < len(pts) - 1 for i in origins)
    kappas = [p[3] for p in pts]
    assert max(kappas) > 0.0
    assert min(kappas) < 0.0


def _s_of(pts: list[tuple[float, float, float, float]]) -> list[float]:
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return s


def _interp_at(pts: list[tuple[float, float, float, float]], s: list[float], s_q: float) -> tuple[float, float, float]:
    total = s[-1]
    s_q = s_q % total if total > 0.0 else 0.0
    lo, hi = 0, len(s) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if s[mid] <= s_q:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    span = s[i + 1] - s[i]
    t = 0.0 if span <= 1e-12 else (s_q - s[i]) / span
    x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
    y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
    yaw = pts[i][2] + t * wrap(pts[i + 1][2] - pts[i][2])
    return x, y, wrap(yaw)


def _closest_in_window(
    pts: list[tuple[float, float, float, float]],
    s_tbl: list[float],
    s_center: float,
    window: float,
    px: float,
    py: float,
    ds: float = 0.05,
) -> float:
    best_s, best_d = s_center, 1e9
    s_try = s_center - window
    while s_try <= s_center + window:
        tx, ty, _ = _interp_at(pts, s_tbl, s_try)
        d = math.hypot(px - tx, py - ty)
        if d < best_d:
            best_d, best_s = d, s_try
        s_try += ds
    return best_s


def _signed_cte(px: float, py: float, tx: float, ty: float, tyaw: float) -> float:
    """Positive when the path is to the left of the point (Stanley convention)."""
    return -math.sin(tyaw) * (tx - px) + math.cos(tyaw) * (ty - py)


def _run_figure8(v: float) -> tuple[list[float], list[float]]:
    pts = make_figure8.load_centerline(CENTERLINE)
    s_tbl = _s_of(pts)
    length = s_tbl[-1]
    ctl = StanleyPID(L_TWIN)
    backend = BicycleBackend(BicycleState(0.0, 0.0, 0.0, v))
    s_front = 0.0
    rear_abs: list[float] = []
    front_abs: list[float] = []
    n_steps = int(length / v / DT) + 1
    for _ in range(n_steps):
        st = backend.state
        fx, fy = st.x + L_TWIN * math.cos(st.yaw), st.y + L_TWIN * math.sin(st.yaw)
        s_front = _closest_in_window(pts, s_tbl, s_front + st.v * DT, 8.0, fx, fy)
        s_front = s_front % length
        tx, ty, tyaw = _interp_at(pts, s_tbl, s_front)
        e = _signed_cte(fx, fy, tx, ty, tyaw)
        delta = ctl.lateral(e, wrap(tyaw - st.yaw), st.v)
        accel = ctl.longitudinal(st.v, v, DT)
        s_rear = _closest_in_window(pts, s_tbl, s_front - L_TWIN, 6.0, st.x, st.y)
        rx, ry, ryaw = _interp_at(pts, s_tbl, s_rear)
        rear_abs.append(abs(_signed_cte(st.x, st.y, rx, ry, ryaw)))
        front_abs.append(abs(e))
        backend.set_command(delta, accel)
        backend.step(DT)
    return rear_abs, front_abs


def test_stanley_tracks_figure8_on_the_twin() -> None:
    rear_abs, front_abs = _run_figure8(V_CRUISE)
    max_rear, max_front = max(rear_abs), max(front_abs)
    rms_rear = math.sqrt(sum(e * e for e in rear_abs) / len(rear_abs))
    rms_front = math.sqrt(sum(e * e for e in front_abs) / len(front_abs))
    print(
        f"figure8 v={V_CRUISE} m/s  max_rear={max_rear:.4f} rms_rear={rms_rear:.4f}  "
        f"max_front={max_front:.4f} rms_front={rms_front:.4f}"
    )
    assert max_rear < LAT_ERR_BAR_M
    assert max_front < LAT_ERR_BAR_M
