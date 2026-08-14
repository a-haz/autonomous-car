"""Watchdog ROS node. Publishes second_miss in BrainHealth.note."""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from autocar_msgs.msg import BrainHealth, SupervisorState
from autocar_watchdog.watchdog import Watchdog

_ST = {0: "INIT", 1: "ARMING", 2: "RUNNING", 3: "DEGRADED", 4: "HOLD", 5: "ESTOP", 6: "DISENGAGED"}


class WatchdogNode(Node):
    def __init__(self) -> None:
        super().__init__("watchdog_node")
        self._wd = Watchdog()
        self._state = "INIT"
        self._episode = 0
        self._ok = {"loc": True, "perc": True, "brain": True, "gate": True}
        self.create_subscription(SupervisorState, "/supervisor/state", self._on_sup, 10)
        self.create_subscription(BrainHealth, "/brains/a/health", self._on_h, 10)
        self.create_subscription(BrainHealth, "/gate/health", self._on_h, 10)
        self._pub = self.create_publisher(BrainHealth, "/watchdog/health", 10)
        self.create_timer(0.02, self._tick)

    def _on_sup(self, msg: SupervisorState) -> None:
        self._state = _ST.get(int(msg.state), "INIT")
        self._episode = int(msg.episode_id)
        self._ok["loc"] = bool(msg.loc_ok)
        self._ok["perc"] = bool(msg.perc_ok)
        self._ok["brain"] = bool(msg.brain_ok)
        self._ok["gate"] = bool(msg.gate_ok)

    def _on_h(self, msg: BrainHealth) -> None:
        if msg.brain == "gate":
            self._ok["gate"] = bool(msg.ok)

    def _tick(self) -> None:
        second = self._wd.tick(
            episode_id=self._episode,
            state=self._state,
            loc_ok=self._ok["loc"],
            perc_ok=self._ok["perc"],
            brain_ok=self._ok["brain"],
            gate_ok=self._ok["gate"],
        )
        msg = BrainHealth()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.brain = "watchdog"
        msg.ok = not second
        msg.degraded = second
        msg.note = "second_miss" if second else ""
        msg.last_ok_stamp = self.get_clock().now().nanoseconds * 1e-9
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = WatchdogNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
