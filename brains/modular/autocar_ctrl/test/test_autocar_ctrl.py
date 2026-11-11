"""PR-04 bicycle + Stanley tests. No ROS."""
from __future__ import annotations

import math

from autocar_ctrl.bicycle import DELTA_MAX, L_TWIN, BicycleParams, BicycleState, bicycle_step, step, wrap
from autocar_ctrl.stanley import StanleyPID


def test_constant_steer_traces_circle_of_radius_L_over_tan_delta() -> None:
    L, delta, v, dt = L_TWIN, 0.3, 5.0, 1e-3
    r = L / math.tan(delta)
    cx, cy = 0.0, r  # left turn from origin heading +x
    s = BicycleState(0.0, 0.0, 0.0, v)
    p = BicycleParams()
    worst = 0.0
    for _ in range(int(2.0 * math.pi * r / v / dt)):
        s = step(s, delta, 0.0, dt, p)
        worst = max(worst, abs(math.hypot(s.x - cx, s.y - cy) - r))
    assert worst < 1e-9
    assert math.hypot(s.x, s.y) < v * dt * 1.01  # back near the start after one lap


def test_exact_arc_quarter_turn_in_one_step() -> None:
    L, delta, v = 2.0, 0.4, 3.0
    r = L / math.tan(delta)
    dt = (math.pi / 2.0) * r / v
    x, y, yaw = bicycle_step(0.0, 0.0, 0.0, v, delta, dt, L)
    assert math.isclose(x, r, abs_tol=1e-9)
    assert math.isclose(y, r, abs_tol=1e-9)
    assert math.isclose(yaw, math.pi / 2.0, abs_tol=1e-12)


def test_step_clamps_steer_and_speed() -> None:
    p = BicycleParams()
    s = step(BicycleState(v=p.v_max), 2.0, 100.0, 0.1, p)
    assert s.v == p.v_max
    ref = step(BicycleState(v=p.v_max), p.delta_max, 0.0, 0.1, p)
    assert math.isclose(s.yaw, ref.yaw, abs_tol=1e-12)
    assert step(BicycleState(v=0.1), 0.0, -100.0, 0.1, p).v == 0.0


def _track_line(y0: float, v: float, steps: int, dt: float = 0.02) -> tuple[list[float], list[float]]:
    """Follow the x axis heading +x. e = -y of the front axle (path is left when y < 0)."""
    L = L_TWIN
    ctl = StanleyPID(L)
    s = BicycleState(0.0, y0, 0.0, v)
    errs, steers = [], []
    for _ in range(steps):
        yf = s.y + L * math.sin(s.yaw)
        e, theta_e = -yf, wrap(0.0 - s.yaw)
        delta = ctl.lateral(e, theta_e, s.v)
        errs.append(abs(e))
        steers.append(delta)
        s = step(s, delta, 0.0, dt)
    return errs, steers


def test_stanley_converges_from_offset_and_respects_limit() -> None:
    errs, steers = _track_line(y0=2.0, v=5.0, steps=1000)
    assert errs[0] > 1.9
    assert errs[-1] < 0.01
    assert max(abs(d) for d in steers) <= DELTA_MAX + 1e-12
    assert min(steers) == -DELTA_MAX  # large offset saturates at the start
    assert steers[0] < 0  # path is to the right, so steer right


def test_stanley_turns_left_onto_gentle_left_arc() -> None:
    R, v, dt, L = 50.0, 5.0, 0.02, L_TWIN
    ctl = StanleyPID(L)
    # Circle centred (0, R), driven CCW, start on it at the origin heading +x.
    s = BicycleState(0.0, 0.0, 0.0, v)
    first = None
    worst = 0.0
    for i in range(1500):
        fx, fy = s.x + L * math.cos(s.yaw), s.y + L * math.sin(s.yaw)
        dx, dy = fx, fy - R
        dist = math.hypot(dx, dy)
        e = dist - R  # outside the circle -> path is to the left
        path_yaw = math.atan2(dy, dx) + math.pi / 2.0
        delta = ctl.lateral(e, wrap(path_yaw - s.yaw), s.v)
        if first is None:
            first = delta
        if i > 500:
            worst = max(worst, abs(e))
        s = step(s, delta, 0.0, dt)
    assert first is not None and first > 0.0
    assert worst < 0.1


def test_longitudinal_clamps() -> None:
    ctl = StanleyPID(L_TWIN)
    assert ctl.longitudinal(0.0, 20.0, 0.02) == ctl.a_max
    assert ctl.longitudinal(20.0, 0.0, 0.02) == ctl.a_min
