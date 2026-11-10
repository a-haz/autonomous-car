"""Supervisor ROS node. Services + /supervisor/state at 50 Hz."""
from __future__ import annotations

import math
from functools import partial

import rclpy
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.node import Node
from std_srvs.srv import Trigger

from autocar_msgs.msg import BrainHealth, SupervisorState, VehicleCommand, VehicleState
from autocar_msgs.srv import Arm
from autocar_supervisor.sm import Brain, Health, Supervisor

_BRAIN_KEY = {Brain.A: "a", Brain.B: "b", Brain.C: "c"}
_REPORTERS = {"a", "b", "c", "gate", "mcu", "loc", "perc"}
# A health report older than this counts as unhealthy (reporters publish at 50 Hz).
HEALTH_STALE_S = 0.100


def _fresh(received_at: float, now: float) -> bool:
    return 0.0 <= now - received_at <= HEALTH_STALE_S


class SupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("supervisor_node")
        self.declare_parameter("active_brain", "A")
        self._sm = Supervisor()
        self._vehicle = (False, False)  # latest (estop_latched, collision)
        self._pulse = (False, False)  # any true since the last tick
        self._reports: dict[str, tuple[float, bool]] = {}  # key -> (received at, ok)
        self._second_miss: tuple[float, bool] | None = None  # owned by the watchdog
        self._cmd_stamp: dict[str, float] = {}
        self._last_now = -math.inf
        # rcl reports a /clock rewind only through this callback; a rewind and
        # return between two clock reads is otherwise invisible.
        self._jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1)),
            post_callback=lambda _jump: self._forget_reports(),
        )
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
        self.create_subscription(VehicleState, "/vehicle/state", self._on_vehicle, 10)
        self._pub = self.create_publisher(SupervisorState, "/supervisor/state", 10)
        self.create_timer(0.02, self._tick)

    def _trig(self, fn):
        def _cb(_req, resp):
            resp.success = fn()
            resp.message = self._sm.reason if resp.success else f"not legal in {self._sm.state.name}"
            return resp

        return _cb

    def _on_arm(self, req: Arm.Request, resp: Arm.Response) -> Arm.Response:
        try:
            brain = Brain(int(req.brain))
        except ValueError:
            resp.ok = False
            resp.message = "bad brain"
            return resp
        resp.ok = self._sm.request_arm(brain)
        resp.message = self._sm.reason if resp.ok else f"not legal in {self._sm.state.name}"
        return resp

    def _now(self) -> float:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_now:  # e.g. a system clock step, which rcl does not report
            self._forget_reports()
        self._last_now = now
        return now

    def _forget_reports(self) -> None:
        """After a backward clock jump nothing received earlier is fresh again,
        however far the clock climbs back. Keys stay, so the boot wait still ends."""
        self._reports = {key: (-math.inf, ok) for key, (_, ok) in self._reports.items()}
        if self._second_miss is not None:
            self._second_miss = (-math.inf, self._second_miss[1])
        self._cmd_stamp.clear()
        self._last_now = -math.inf

    def _on_health(self, msg: BrainHealth) -> None:
        key = msg.brain.lower()
        now = self._now()
        if key == "watchdog":
            self._second_miss = (now, "second_miss" in msg.note)
        elif key in _REPORTERS:
            self._reports[key] = (now, bool(msg.ok))

    def _ok(self, key: str | None, now: float, unheard: bool = False) -> bool:
        """Latest report from key is ok and fresh; ``unheard`` if it never reported."""
        report = self._reports.get(key) if key else None
        if report is None:
            return unheard
        return report[1] and _fresh(report[0], now)

    def _on_cmd(self, key: str, msg: VehicleCommand) -> None:
        self._cmd_stamp[key] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_vehicle(self, msg: VehicleState) -> None:
        self._vehicle = (bool(msg.estop_latched), bool(msg.collision))
        # Keep a pulse until a step reads it; the state machine latches ESTOP.
        self._pulse = (self._pulse[0] or self._vehicle[0], self._pulse[1] or self._vehicle[1])

    def _tick(self) -> None:
        now = self._now()
        stamp = self._cmd_stamp.get(_BRAIN_KEY.get(self._sm.brain, ""))
        second = self._second_miss
        h = Health(
            odd=True,  # the ODD checklist is not evaluated yet
            loc=self._ok("loc", now, unheard=True),  # loc/perc have no reporter yet
            perc=self._ok("perc", now, unheard=True),
            brain=self._ok(_BRAIN_KEY.get(self._sm.brain), now),
            mcu=self._ok("mcu", now),
            gate=self._ok("gate", now),
            e_stop=self._vehicle[0] or self._pulse[0],
            collision=self._vehicle[1] or self._pulse[1],
            second_miss=second is None or not _fresh(second[0], now) or second[1],
            cmd_age_s=math.inf if stamp is None else abs(now - stamp),
        )
        # Hold INIT until the MCU has reported once: stepping with mcu=false
        # would latch ESTOP on a node start-up race. A pulse seen while waiting
        # is kept for the first step, so an e-stop or collision is never lost.
        if "mcu" in self._reports:
            self._sm.step(h)
            self._pulse = (False, False)
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
