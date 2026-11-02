"""Supervisor ROS node. Services + /supervisor/state at 50 Hz."""
from __future__ import annotations

import math
from functools import partial

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from autocar_msgs.msg import BrainHealth, SupervisorState, VehicleCommand
from autocar_msgs.srv import Arm
from autocar_supervisor.sm import Brain, Health, St, Supervisor

_BRAIN_KEY = {Brain.A: "a", Brain.B: "b", Brain.C: "c"}
_REPORTERS = {"a", "b", "c", "gate", "mcu", "loc", "perc"}
# A health report older than this counts as unhealthy (reporters publish at 50 Hz).
HEALTH_STALE_S = 0.100


class SupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("supervisor_node")
        self.declare_parameter("active_brain", "A")
        self._sm = Supervisor()
        self._health = {"e_stop": False, "collision": False}
        self._reports: dict[str, tuple[float, bool]] = {}  # key -> (received at, ok)
        self._second_miss: tuple[float, bool] | None = None  # owned by the watchdog
        self._cmd_stamp: dict[str, float] = {}
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
        for key in ("a", "b", "c"):
            self.create_subscription(VehicleCommand, f"/brains/{key}/command", partial(self._on_cmd, key), 10)
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
        now = self.get_clock().now().nanoseconds * 1e-9
        if key == "watchdog":
            self._second_miss = (now, "second_miss" in msg.note)
        elif key in _REPORTERS:
            self._reports[key] = (now, bool(msg.ok))

    def _ok(self, key: str | None, now: float, unheard: bool = False) -> bool:
        """Latest report from key is ok and fresh; ``unheard`` if it never reported."""
        report = self._reports.get(key) if key else None
        if report is None:
            return unheard
        return report[1] and now - report[0] <= HEALTH_STALE_S

    def _on_cmd(self, key: str, msg: VehicleCommand) -> None:
        self._cmd_stamp[key] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        stamp = self._cmd_stamp.get(_BRAIN_KEY.get(self._sm.brain, ""))
        second = self._second_miss
        h = Health(
            odd=True,  # the ODD checklist is not evaluated yet
            loc=self._ok("loc", now, unheard=True),  # loc/perc have no reporter yet
            perc=self._ok("perc", now, unheard=True),
            brain=self._ok(_BRAIN_KEY.get(self._sm.brain), now),
            mcu=self._ok("mcu", now),
            gate=self._ok("gate", now),
            e_stop=self._health["e_stop"],
            collision=self._health["collision"],
            second_miss=second is None or now - second[0] > HEALTH_STALE_S or second[1],
            cmd_age_s=math.inf if stamp is None else abs(now - stamp),
        )
        # Hold INIT until the MCU has reported once: stepping with mcu=false
        # would latch ESTOP on a node start-up race.
        if "mcu" in self._reports:
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
