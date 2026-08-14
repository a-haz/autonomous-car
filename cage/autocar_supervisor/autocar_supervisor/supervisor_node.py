"""Supervisor ROS node. Services + /supervisor/state at 50 Hz."""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from autocar_msgs.msg import BrainHealth, SupervisorState
from autocar_msgs.srv import Arm
from autocar_supervisor.sm import Brain, Health, St, Supervisor


class SupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("supervisor_node")
        self.declare_parameter("use_sim_time", True)
        self.declare_parameter("active_brain", "A")
        self._sm = Supervisor()
        self._health = {
            "odd": True,
            "loc": True,
            "perc": True,
            "brain": False,
            "mcu": True,
            "gate": True,
            "e_stop": False,
            "collision": False,
        }
        self._second_miss = False
        self._cmd_age = math.inf
        self._last_cmd_stamp: float | None = None
        self.create_service(Arm, "/supervisor/arm", self._on_arm)
        self.create_service(Trigger, "/supervisor/hold", self._trig(self._sm.request_hold))
        self.create_service(Trigger, "/supervisor/resume", self._trig(self._sm.request_resume))
        self.create_service(Trigger, "/supervisor/disengage", self._trig(self._sm.request_disengage))
        self.create_service(Trigger, "/supervisor/reset", self._trig(self._sm.request_reset))
        self.create_subscription(BrainHealth, "/brains/a/health", self._on_health, 10)
        self.create_subscription(BrainHealth, "/brains/b/health", self._on_health, 10)
        self.create_subscription(BrainHealth, "/brains/c/health", self._on_health, 10)
        self.create_subscription(BrainHealth, "/gate/health", self._on_health, 10)
        self.create_subscription(BrainHealth, "/mcu/health", self._on_health, 10)
        self.create_subscription(BrainHealth, "/watchdog/health", self._on_health, 10)
        self._pub = self.create_publisher(SupervisorState, "/supervisor/state", 10)
        self.create_timer(0.02, self._tick)

    def _trig(self, fn):
        def _cb(_req, resp):
            fn()
            resp.success = True
            resp.message = self._sm.reason
            return resp

        return _cb

    def _on_arm(self, req: Arm.Request, resp: Arm.Response) -> Arm.Response:
        try:
            brain = Brain(int(req.brain))
        except ValueError:
            resp.ok = False
            resp.message = "bad brain"
            return resp
        self._sm.request_arm(brain)
        resp.ok = True
        resp.message = self._sm.reason
        return resp

    def _on_health(self, msg: BrainHealth) -> None:
        key = msg.brain.lower()
        ok = bool(msg.ok)
        if key in {"a", "b", "c"}:
            armed = {1: "a", 2: "b", 3: "c"}.get(int(self._sm.brain), "")
            if key == armed:
                self._health["brain"] = ok
        elif key == "gate":
            self._health["gate"] = ok
        elif key == "loc":
            self._health["loc"] = ok
        elif key == "perc":
            self._health["perc"] = ok
        elif key == "mcu":
            self._health["mcu"] = ok
        elif key == "watchdog":
            pass
        if "second_miss" in msg.note:
            self._second_miss = True

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        h = Health(
            odd=self._health["odd"],
            loc=self._health["loc"],
            perc=self._health["perc"],
            brain=self._health["brain"],
            mcu=self._health["mcu"],
            gate=self._health["gate"],
            e_stop=self._health["e_stop"],
            collision=self._health["collision"],
            second_miss=self._second_miss,
            cmd_age_s=self._cmd_age,
        )
        self._sm.step(h)
        out = SupervisorState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.state = int(self._sm.state)
        out.active_brain = int(self._sm.brain)
        out.odd_ok = h.odd
        out.loc_ok = h.loc
        out.perc_ok = h.perc
        out.brain_ok = h.brain
        out.mcu_ok = h.mcu
        out.gate_ok = h.gate
        out.reason = self._sm.reason
        out.episode_id = self._sm.episode_id
        self._pub.publish(out)
        _ = now


def main() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
