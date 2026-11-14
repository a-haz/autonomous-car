"""PR-05 G1 centerline planner unit tests. No ROS."""
from __future__ import annotations

import math

import numpy as np
from autocar_perc_gt.stub import CLASS_CAR, CLASS_STATIC, LIGHT_GREEN, LIGHT_RED, GtObj, to_tracked
from autocar_plan.simple_planner import (
    DT,
    HALF_W,
    N,
    R_INFLATE,
    TRACK_M,
    CenterlinePlanner,
    closest_frenet,
    light_is_red,
    plan,
    resample_centerline,
    static_xy,
)
from autocar_command_gate.gate import GateParams

LINE = np.stack([np.linspace(0.0, 100.0, 401), np.zeros(401)], axis=1)
V = 5.0


def test_params_match_gate_and_design() -> None:
    p = GateParams()
    assert TRACK_M == p.track_m and R_INFLATE == p.r_inflate
    assert HALF_W == TRACK_M / 2 + R_INFLATE


def test_resample_is_64_by_0p1s() -> None:
    traj = resample_centerline(LINE, V)
    assert traj.shape == (N, 5)
    assert np.allclose(np.diff(traj[:, 0]), DT)
    assert np.allclose(np.diff(traj[:, 1]), V * DT)
    assert np.allclose(traj[:, 2], 0.0) and np.allclose(traj[:, 3], 0.0)
    assert np.allclose(traj[:, 4], V)


def test_static_in_corridor_stops_and_outside_does_not() -> None:
    inside = np.array([[20.0, HALF_W - 0.01]])
    outside = np.array([[20.0, HALF_W + 0.01]])
    assert np.all(plan(LINE, V, inside, False)[:, 4] == 0.0)
    assert np.all(plan(LINE, V, outside, False)[:, 4] == V)


def test_moving_actor_in_corridor_does_not_zero_v() -> None:
    tracked = to_tracked(0, [GtObj(1, CLASS_CAR, 10.0, 0.0, 0.0, 4.5, 1.9, 1.6, vx=3.0)])
    assert len(static_xy(tracked)) == 0
    assert np.all(plan(LINE, V, static_xy(tracked), False)[:, 4] == V)


def test_static_class_and_stopped_car_count_as_static() -> None:
    tracked = to_tracked(0, [GtObj(1, CLASS_STATIC, 10.0, 0.0, 0.0, 1, 1, 1),
                             GtObj(2, CLASS_CAR, 30.0, 0.0, 0.0, 4.5, 1.9, 1.6)])
    assert static_xy(tracked).shape == (2, 2)


def test_red_binds_to_own_lanelet_only() -> None:
    lights_own = {"lights": [{"lanelet_id": 11, "light_state": LIGHT_RED}]}
    lights_other = {"lights": [{"lanelet_id": 12, "light_state": LIGHT_RED},
                               {"lanelet_id": 11, "light_state": LIGHT_GREEN}]}
    assert light_is_red(lights_own, 11)
    assert not light_is_red(lights_other, 11)
    assert not light_is_red({"lights": []}, 11)
    pl = CenterlinePlanner(LINE, 11, V, closed=False)
    empty = to_tracked(0, [])
    assert np.all(pl.step((0.0, 0.0, 0.0), empty, lights_own)[:, 4] == 0.0)
    assert np.all(pl.step((0.0, 0.0, 0.0), empty, lights_other)[:, 4] == V)


def test_stop_keeps_geometry_for_the_controller() -> None:
    traj = plan(LINE, V, np.zeros((0, 2)), True)
    assert np.allclose(np.diff(traj[:, 1]), V * DT)
    e, th, vref, _ = closest_frenet((3.0, 0.4, 0.1), traj)
    assert math.isclose(e, -0.4, abs_tol=1e-9) and math.isclose(th, -0.1, abs_tol=1e-9) and vref == 0.0


def test_closest_frenet_sign_left_positive() -> None:
    traj = resample_centerline(LINE, V)
    e, th, vref, yaw = closest_frenet((5.25, -0.7, 0.0), traj)
    assert math.isclose(e, 0.7, abs_tol=1e-9) and th == 0.0 and vref == V and yaw == 0.0


def test_planner_cannot_tell_gt_from_hand_built() -> None:
    gt = to_tracked(5, [GtObj(3, CLASS_STATIC, 15.0, 0.2, 0.0, 1, 1, 1)])
    hand = {"stamp_ns": 5, "frame_id": "map", "objects": [dict(gt["objects"][0])]}
    a = CenterlinePlanner(LINE, 1, V, closed=False).step((0.0, 0.0, 0.0), gt, {})
    b = CenterlinePlanner(LINE, 1, V, closed=False).step((0.0, 0.0, 0.0), hand, {})
    assert np.array_equal(a, b)
