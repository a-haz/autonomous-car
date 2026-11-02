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


def _feed(ros, node, t: float, *, brain: str | None = "a", cmd_brain: str | None = "a", mcu: bool = True) -> None:
    """One 50 Hz round of inputs, received at time t."""
    node.now_ns = round(t * 1e9)
    ros.deliver(node, "/mcu/health", _health(ros, "mcu", mcu))
    ros.deliver(node, "/gate/health", _health(ros, "gate"))
    ros.deliver(node, "/watchdog/health", _health(ros, "watchdog"))
    if brain:
        ros.deliver(node, f"/brains/{brain}/health", _health(ros, brain.upper()))
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
