"""PR-02 gate tests. No ROS."""
from __future__ import annotations

import numpy as np

from autocar_command_gate.gate import (
    Cmd,
    CommandGate,
    GateParams,
    Lead,
    occupancy_hit,
    rss_dmin,
    select_lead,
    slice_to_xy,
)


def _healthy_cmd(now: float, steer: float = 0.0, accel: float = 1.0) -> Cmd:
    return Cmd(now, steer, accel, False, 1, "A")


def test_rss_uses_two_decelerations() -> None:
    p = GateParams(a_min_brake_rear=6.0, a_max_brake_front=8.0)
    two = rss_dmin(10.0, 10.0, p)
    same = rss_dmin(10.0, 10.0, GateParams(a_min_brake_rear=6.0, a_max_brake_front=6.0))
    assert two > same
    v_r, v_f = 10.0, 10.0
    rho, a_max_a = p.rho_s, p.a_max_accel
    expect = v_r * rho + 0.5 * a_max_a * rho * rho
    expect += (v_r + rho * a_max_a) ** 2 / (2.0 * 6.0)
    expect -= (v_f * v_f) / (2.0 * 8.0)
    assert abs(two - expect) < 1e-9


def test_degraded_one_tick_command_shape() -> None:
    p = GateParams()
    g = CommandGate(p)
    g._last = Cmd(0.0, 0.12, 0.0, False, 0, "gate")
    now = 1.0
    incoming = _healthy_cmd(now, steer=0.4, accel=2.0)
    out = g.tick(now, incoming, "DEGRADED", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 11.0, False)
    assert out.reason == "degraded"
    assert out.verdict == "clip"
    assert abs(out.cmd.steer - 0.12) < 1e-9
    j = p.jerk_max * p.dt
    assert abs(out.cmd.accel + j) < 1e-6
    assert out.cmd.accel != -p.a_hold


def test_degraded_settle_after_117_or_3s() -> None:
    p = GateParams()
    g = CommandGate(p)
    v = 11.0
    n = 150
    for i in range(n):
        now = i * p.dt
        incoming = _healthy_cmd(now, accel=2.0)
        out = g.tick(now, incoming, "DEGRADED", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), v, False)
        v = max(0.0, v + out.cmd.accel * p.dt)
    assert v <= p.v_degrade + 1e-3


def test_rss_hold_then_clip_recheck() -> None:
    p = GateParams()
    g = CommandGate(p)
    dmin = rss_dmin(10.0, 0.0, p)
    lead = Lead(dx=dmin * 0.5, v_rel=-10.0, v_ego=10.0, v_lead=0.0)
    out = g.tick(0.0, _healthy_cmd(0.0), "RUNNING", lead, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 10.0, False)
    assert out.verdict == "hold"
    assert out.reason == "rss"
    assert abs(out.cmd.accel + p.a_min_brake_rear) < 1e-9


def test_stale_is_comfort_hold() -> None:
    g = CommandGate()
    out = g.tick(1.0, _healthy_cmd(0.0), "RUNNING", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 5.0, False)
    assert out.verdict == "hold"
    assert out.reason == "stale"
    assert abs(out.cmd.accel + g.p.a_hold) < 1e-9


def test_nan_or_future_stamp_is_stale() -> None:
    nan = float("nan")
    for now, stamp in ((100.0, nan), (10.0, 1.7e9), (500.0, 1.7e9)):
        g = CommandGate()
        out = g.tick(now, _healthy_cmd(stamp), "RUNNING", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 5.0, False)
        assert (out.verdict, out.reason) == ("hold", "stale")
    g = CommandGate()
    out = g.tick(1.0, _healthy_cmd(1.05), "RUNNING", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 5.0, False)
    assert out.verdict != "hold"


def test_nan_command_is_held_and_does_not_stick() -> None:
    nan = float("nan")
    occ = np.zeros((0, 2), np.float32)
    g = CommandGate()
    for steer, accel in ((nan, 0.5), (0.0, nan), (float("inf"), 0.5)):
        out = g.tick(0.0, _healthy_cmd(0.0, steer=steer, accel=accel), "RUNNING", None, occ, (0.0, 0.0, 0.0), 5.0, False)
        assert (out.verdict, out.reason) == ("hold", "invalid")
        assert np.isfinite([out.cmd.steer, out.cmd.accel]).all()
    out = g.tick(0.02, _healthy_cmd(0.02), "RUNNING", None, occ, (0.0, 0.0, 0.0), 5.0, False)
    assert out.verdict != "hold"
    assert np.isfinite([out.cmd.steer, out.cmd.accel]).all()


def test_non_finite_ego_state_holds() -> None:
    nan = float("nan")
    occ = np.zeros((0, 2), np.float32)
    lead = Lead(dx=1.0, v_rel=-10.0, v_ego=nan, v_lead=0.0)
    for v, pose in ((nan, (0.0, 0.0, 0.0)), (5.0, (nan, 0.0, 0.0)), (5.0, (0.0, 0.0, nan))):
        out = CommandGate().tick(0.0, _healthy_cmd(0.0), "RUNNING", lead, occ, pose, v, False)
        assert out.verdict == "hold"


def test_accel_bound_is_reported_as_clip() -> None:
    p = GateParams()
    g = CommandGate(p)
    g._last = Cmd(0.0, 0.0, p.a_max_accel, False, 0, "A")
    ask = p.a_max_accel + 0.5 * p.jerk_max * p.dt
    out = g.tick(0.0, _healthy_cmd(0.0, accel=ask), "RUNNING", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 5.0, False)
    assert abs(out.cmd.accel - p.a_max_accel) < 1e-9
    assert (out.verdict, out.reason) == ("clip", "clip")
    g = CommandGate(p)
    out = g.tick(0.0, _healthy_cmd(0.0, accel=0.05), "RUNNING", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 5.0, False)
    assert (out.verdict, out.reason) == ("pass", "ok")


def test_hold_seq_wraps_at_uint32() -> None:
    occ = np.zeros((0, 2), np.float32)
    g = CommandGate()
    g.tick(0.0, Cmd(0.0, 0.0, 0.0, False, 0xFFFFFFFF, "A"), "RUNNING", None, occ, (0.0, 0.0, 0.0), 0.0, False)
    out = g.tick(1.0, None, "RUNNING", None, occ, (0.0, 0.0, 0.0), 0.0, False)
    assert out.verdict == "hold"
    assert out.cmd.seq == 0


def test_none_brain_supervisor_is_comfort() -> None:
    g = CommandGate()
    out = g.tick(0.0, _healthy_cmd(0.0), "INIT", None, np.zeros((0, 2), np.float32), (0.0, 0.0, 0.0), 0.0, False)
    assert out.verdict == "hold"
    assert out.reason == "supervisor"


def test_occupancy_hit_epsilon_only() -> None:
    p = GateParams(inflate_m=0.05)
    occ = np.array([[0.0, 0.0]], dtype=np.float32)
    assert occupancy_hit(occ, (0.0, 0.0, 0.0), 8.0, p)


def test_select_lead_ignores_static() -> None:
    p = GateParams()
    lead = select_lead([(6, 8.0, 0.0), (1, 12.0, 0.0)], (0.0, 0.0, 0.0), 8.0, p, [0.0, 0.0])
    assert lead is not None
    assert abs(lead.dx - 12.0) < 1e-6


def test_slice_to_xy_centroids() -> None:
    class S:
        width = 4
        height = 3
        resolution = 1.0
        origin_x = 0.0
        origin_y = 0.0
        cells = [0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0]

    xy = slice_to_xy(S())
    assert xy.shape == (1, 2)
    assert abs(xy[0, 0] - 1.5) < 1e-5
    assert abs(xy[0, 1] - 1.5) < 1e-5
