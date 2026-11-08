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
        self._second = False
        self.create_subscription(SupervisorState, "/supervisor/state", self._on_sup, 10)
        self._pub = self.create_publisher(BrainHealth, "/watchdog/health", 10)
        self.create_timer(0.02, self._tick)

    def _on_sup(self, msg: SupervisorState) -> None:
        # One evaluation per supervisor sample, on the flags the supervisor stepped
        # on. Evaluating on our own timer counted a sample twice or skipped one.
        self._second = self._wd.tick(
            episode_id=int(msg.episode_id),
            state=_ST.get(int(msg.state), "INIT"),
            loc_ok=bool(msg.loc_ok),
            perc_ok=bool(msg.perc_ok),
            brain_ok=bool(msg.brain_ok),
            gate_ok=bool(msg.gate_ok),
        )

    def _tick(self) -> None:
        second = self._second
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
