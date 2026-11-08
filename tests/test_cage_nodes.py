"""Cage node wiring under the rclpy stand-in in conftest.py. No ROS 2 required."""

from __future__ import annotations

import numpy as np
import pytest

from autocar_occ.slice import from_lidar_xy

NODES = (
    ("autocar_command_gate.gate_node", "GateNode", "/vehicle/command"),
    ("autocar_supervisor.supervisor_node", "SupervisorNode", "/supervisor/state"),
    ("autocar_watchdog.watchdog_node", "WatchdogNode", "/watchdog/health"),
    ("autocar_mcu_bridge.mcu_node", "McuNode", "/mcu/health"),
    ("autocar_occ.occ_node", "OccNode", None),
)


@pytest.mark.parametrize(("module", "cls", "topic"), NODES)
def test_node_starts_and_ticks(ros, module: str, cls: str, topic: str | None) -> None:
    node = ros.make(module, cls)
    ros.spin_once(node, 1.0)
    if topic is not None:
        assert node.sent[topic]


SUP = ("autocar_supervisor.supervisor_node", "SupervisorNode")
ST_ARMING, ST_RUNNING, ST_DEGRADED, ST_HOLD, ST_ESTOP = 1, 2, 3, 4, 5


def _health(ros, name: str, ok: bool = True, note: str = ""):
    return ros.Msg(brain=name, ok=ok, note=note)


def _cmd(ros, t: float):
    return ros.Msg(
        header=ros.Msg(stamp=ros.stamp(t)),
        steering_tire_angle_rad=0.0,
        accel_mps2=0.5,
        hold=False,
        seq=1,
        source="A",
    )


def _feed(
    ros,
    node,
    t: float,
    *,
    brain: str | None = "a",
    brain_note: str = "",
    cmd_brain: str | None = "a",
    mcu: bool | None = True,
    gate: bool | None = True,
    watchdog: str | None = "",
) -> None:
    """One 50 Hz round of inputs received at time t. None = that input is silent."""
    node.now_ns = round(t * 1e9)
    if mcu is not None:
        ros.deliver(node, "/mcu/health", _health(ros, "mcu", mcu))
    if gate is not None:
        ros.deliver(node, "/gate/health", _health(ros, "gate", gate))
    if watchdog is not None:
        ros.deliver(node, "/watchdog/health", _health(ros, "watchdog", not watchdog, watchdog))
    if brain:
        ros.deliver(node, f"/brains/{brain}/health", _health(ros, brain.upper(), note=brain_note))
    if cmd_brain:
        ros.deliver(node, f"/brains/{cmd_brain}/command", _cmd(ros, t))


def _arm(ros, node, brain: int = 1) -> None:
    node.services["/supervisor/arm"](ros.Msg(brain=brain), ros.Msg())


def _run(ros, node, t0: float, n: int, **feed) -> list[int]:
    states = []
    for k in range(n):
        t = t0 + 0.02 * k
        _feed(ros, node, t, **feed)
        ros.spin_once(node, t)
        states.append(ros.last(node, "/supervisor/state").state)
    return states


def test_supervisor_reaches_running_on_the_armed_brains_commands(ros) -> None:
    node = ros.make(*SUP)
    _arm(ros, node)
    states = _run(ros, node, 1.0, 5)
    assert states[0] == ST_ARMING
    assert states[-1] == ST_RUNNING


def test_supervisor_ignores_commands_from_other_brains(ros) -> None:
    node = ros.make(*SUP)
    _arm(ros, node)
    assert _run(ros, node, 1.0, 10, cmd_brain="b")[-1] == ST_ARMING


def _running(ros, t0: float = 1.0):
    node = ros.make(*SUP)
    _arm(ros, node)
    assert _run(ros, node, t0, 5)[-1] == ST_RUNNING
    return node


def test_supervisor_waits_for_the_mcu_before_arming(ros) -> None:
    node = ros.make(*SUP)
    _arm(ros, node)
    assert _run(ros, node, 1.0, 10, mcu=None) == [0] * 10  # INIT: neither ARMING nor a latched ESTOP
    assert _run(ros, node, 1.2, 5)[-1] == ST_RUNNING


def test_supervisor_silent_mcu_is_estop(ros) -> None:
    node = _running(ros)
    assert _run(ros, node, 1.1, 10, mcu=None)[-1] == ST_ESTOP


def test_supervisor_silent_gate_degrades(ros) -> None:
    node = _running(ros)
    assert ST_DEGRADED in _run(ros, node, 1.1, 10, gate=None)


def test_supervisor_silent_brain_degrades_then_holds(ros) -> None:
    node = _running(ros)
    states = _run(ros, node, 1.1, 40, brain=None)
    assert ST_DEGRADED in states
    assert states[-1] == ST_HOLD


def test_supervisor_new_brain_does_not_inherit_health(ros) -> None:
    node = _running(ros)
    node.services["/supervisor/hold"](ros.Msg(), ros.Msg())
    assert _run(ros, node, 1.1, 2)[-1] == ST_HOLD
    _arm(ros, node, brain=2)
    # B's commands arrive, B never reports health, A keeps reporting.
    assert _run(ros, node, 1.2, 10, cmd_brain="b")[-1] == ST_ARMING


def test_supervisor_second_miss_comes_only_from_the_watchdog(ros) -> None:
    node = _running(ros)
    _run(ros, node, 1.1, 3, brain_note="second_miss")
    assert _run(ros, node, 1.2, 1, gate=False) == [ST_DEGRADED]
    states = _run(ros, node, 1.22, 60)
    assert ST_HOLD not in states
    assert states[-1] == ST_RUNNING


def test_supervisor_second_miss_clears_with_the_watchdog(ros) -> None:
    node = _running(ros)
    assert _run(ros, node, 1.1, 2, gate=False, watchdog="second_miss")[-1] == ST_HOLD
    node.services["/supervisor/disengage"](ros.Msg(), ros.Msg())
    _run(ros, node, 1.2, 1)
    node.services["/supervisor/reset"](ros.Msg(), ros.Msg())
    _run(ros, node, 1.3, 1)
    _arm(ros, node)
    assert _run(ros, node, 1.4, 5)[-1] == ST_RUNNING
    assert _run(ros, node, 1.5, 1, gate=False) == [ST_DEGRADED]
    assert ST_HOLD not in _run(ros, node, 1.52, 5)


def test_supervisor_silent_watchdog_counts_as_second_miss(ros) -> None:
    node = _running(ros)
    _run(ros, node, 1.1, 10, watchdog=None)
    assert _run(ros, node, 1.3, 2, gate=False, watchdog=None) == [ST_DEGRADED, ST_HOLD]


def test_supervisor_latches_estop_on_vehicle_collision_or_estop(ros) -> None:
    for field in ("collision", "estop_latched"):
        node = _running(ros)
        state = ros.Msg(collision=False, estop_latched=False)
        setattr(state, field, True)
        ros.deliver(node, "/vehicle/state", state)
        ros.deliver(node, "/vehicle/state", ros.Msg(collision=False, estop_latched=False))
        assert _run(ros, node, 1.1, 3) == [ST_ESTOP] * 3, field


GATE = ("autocar_command_gate.gate_node", "GateNode")
VERDICT_HOLD = 2


def _gate_round(ros, node, t: float, *, sup: bool = True, active: int = 1, cmd_brain: str | None = "a"):
    node.now_ns = round(t * 1e9)
    if sup:
        ros.deliver(node, "/supervisor/state", ros.Msg(state=ST_RUNNING, active_brain=active))
    if cmd_brain:
        ros.deliver(node, f"/brains/{cmd_brain}/command", _cmd(ros, t))
    ros.spin_once(node, t)
    status = ros.last(node, "/gate/status")
    return status.verdict, status.reason


def test_gate_takes_commands_only_from_the_armed_brain(ros) -> None:
    node = ros.make(*GATE)
    assert _gate_round(ros, node, 1.0)[0] != VERDICT_HOLD
    assert _gate_round(ros, node, 1.02, cmd_brain="b") != (VERDICT_HOLD, "stale")  # A's 1.0 command is still fresh
    assert _gate_round(ros, node, 1.2, cmd_brain="b") == (VERDICT_HOLD, "stale")
    # switching the armed brain drops the old brain's last command at once
    assert _gate_round(ros, node, 1.3)[0] != VERDICT_HOLD
    assert _gate_round(ros, node, 1.31, active=2, cmd_brain=None) == (VERDICT_HOLD, "stale")
    assert _gate_round(ros, node, 1.32, active=2, cmd_brain="b")[0] != VERDICT_HOLD


def test_gate_holds_when_the_supervisor_goes_silent(ros) -> None:
    node = ros.make(*GATE)
    assert _gate_round(ros, node, 1.0)[0] != VERDICT_HOLD
    verdicts = [_gate_round(ros, node, 1.0 + 0.02 * k, sup=False) for k in range(1, 10)]
    assert verdicts[0][0] != VERDICT_HOLD
    assert verdicts[-1] == (VERDICT_HOLD, "supervisor")


def test_gate_holds_for_a_slice_obstacle_on_a_cell_edge_path(ros) -> None:
    # Default slice geometry puts y = 0 on a cell edge, 0.05 m from every centroid.
    node = ros.make(*GATE)
    sl = from_lidar_xy(np.array([[5.0, 0.0]], np.float32), r_inflate=0.30, track=1.55)
    ros.deliver(node, "/perception/occupancy_slice", ros.Msg(
        width=sl.width, height=sl.height, resolution=sl.resolution,
        origin_x=sl.origin_x, origin_y=sl.origin_y, cells=sl.cells.flatten().tolist(),
    ))
    pose = ros.Msg(position=ros.Msg(x=0.0, y=0.0), orientation=ros.Msg(x=0.0, y=0.0, z=0.0, w=1.0))
    ros.deliver(node, "/localization/pose", ros.Msg(pose=ros.Msg(pose=pose)))
    ros.deliver(node, "/vehicle/state", ros.Msg(v_mps=11.0, collision=False))
    assert _gate_round(ros, node, 1.0) == (VERDICT_HOLD, "occupancy")


WD = ("autocar_watchdog.watchdog_node", "WatchdogNode")


def _sup_state(ros, state: int, loc_ok: bool, episode: int = 0):
    return ros.Msg(state=state, episode_id=episode, loc_ok=loc_ok, perc_ok=True, brain_ok=True, gate_ok=True)


def test_watchdog_counts_each_supervisor_sample_once(ros) -> None:
    node = ros.make(*WD)
    ros.deliver(node, "/supervisor/state", _sup_state(ros, ST_RUNNING, True))
    ros.deliver(node, "/supervisor/state", _sup_state(ros, ST_DEGRADED, False))
    for k in range(5):  # several timer periods on the same sample (e.g. a long /clock step)
        ros.spin_once(node, 1.0 + 0.02 * k)
    assert ros.last(node, "/watchdog/health").note == ""
    ros.deliver(node, "/supervisor/state", _sup_state(ros, ST_DEGRADED, False))
    ros.spin_once(node, 1.2)
    assert ros.last(node, "/watchdog/health").note == "second_miss"
    ros.deliver(node, "/supervisor/state", _sup_state(ros, 0, True, episode=1))
    ros.spin_once(node, 1.22)
    assert ros.last(node, "/watchdog/health").note == ""


def test_mcu_node_refuses_a_backend_it_does_not_have(ros) -> None:
    with pytest.raises(ValueError, match="uart"):
        ros.make("autocar_mcu_bridge.mcu_node", "McuNode", backend="uart")
    node = ros.make("autocar_mcu_bridge.mcu_node", "McuNode", backend="loopback")
    ros.spin_once(node, 1.0)
    assert ros.last(node, "/mcu/health").ok is True


OCC = ("autocar_occ.occ_node", "OccNode")
POINTS = [(1.0, 2.0, 0.4), (-3.0, 0.5, 0.3), (2.0, -2.0, 5.0), (float("nan"), 1.0, 0.4)]


def _cloud(ros, *, fmt: str = "f4", big: bool = False, pad: int = 0, datatype: int | None = None):
    order = ">" if big else "<"
    item = np.dtype([("x", order + fmt), ("y", order + fmt), ("z", order + fmt), ("i", order + "f4")])
    arr = np.zeros(len(POINTS), item)
    arr["x"], arr["y"], arr["z"] = np.asarray(POINTS).T
    rows = arr.reshape(2, 2).view(np.uint8).reshape(2, -1)
    rows = np.hstack([rows, np.full((2, pad), 0, np.uint8)])
    dt = datatype if datatype is not None else {"f4": 7, "f8": 8}[fmt]
    fields = [ros.Msg(name=n, offset=item.fields[n][1], datatype=dt if n != "i" else 7, count=1) for n in "xyzi"]
    return ros.Msg(
        header=ros.Msg(stamp=ros.stamp(1.0), frame_id="map"),
        height=2,
        width=2,
        fields=fields,
        is_bigendian=big,
        point_step=item.itemsize,
        row_step=2 * item.itemsize + pad,
        data=rows.tobytes(),
        is_dense=False,
    )


def test_occ_node_parses_padded_big_endian_and_double_clouds(ros) -> None:
    expected = from_lidar_xy(np.array([[1.0, 2.0], [-3.0, 0.5]], np.float32), r_inflate=0.0, track=0.0)
    for kw in ({}, {"pad": 8}, {"big": True}, {"fmt": "f8", "pad": 4}):
        node = ros.make(*OCC, r_inflate=0.0, track=0.0)
        ros.deliver(node, "/sensors/lidar/points", _cloud(ros, **kw))
        out = ros.last(node, "/perception/occupancy_slice")
        assert out.cells == expected.cells.flatten().tolist(), kw


def test_occ_node_drops_a_cloud_it_cannot_read(ros) -> None:
    node = ros.make(*OCC)
    ros.deliver(node, "/sensors/lidar/points", _cloud(ros, datatype=2))
    assert node.sent["/perception/occupancy_slice"] == []
    assert node.logs and node.logs[-1][0] == "error"


def test_supervisor_services_refuse_illegal_requests(ros) -> None:
    node = _running(ros)
    for name in ("/supervisor/reset", "/supervisor/resume"):
        resp = node.services[name](ros.Msg(), ros.Msg())
        assert resp.success is False
        assert resp.message == "not legal in RUNNING"
    assert node.services["/supervisor/arm"](ros.Msg(brain=2), ros.Msg()).ok is False
    assert node.services["/supervisor/arm"](ros.Msg(brain=0), ros.Msg()).ok is False
    assert node.services["/supervisor/hold"](ros.Msg(), ros.Msg()).success is True


def test_supervisor_ignores_an_estop_released_before_it_starts_stepping(ros) -> None:
    node = ros.make(*SUP)
    ros.deliver(node, "/vehicle/state", ros.Msg(collision=False, estop_latched=True))
    ros.spin_once(node, 0.5)  # no MCU report yet: not stepping
    ros.deliver(node, "/vehicle/state", ros.Msg(collision=False, estop_latched=False))
    _arm(ros, node)
    assert _run(ros, node, 1.0, 5)[-1] == ST_RUNNING
