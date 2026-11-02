"""Cage node wiring under the rclpy stand-in in conftest.py. No ROS 2 required."""

from __future__ import annotations

import pytest

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
