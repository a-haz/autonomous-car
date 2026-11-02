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
