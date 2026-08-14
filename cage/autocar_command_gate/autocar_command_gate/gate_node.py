"""Command gate ROS node. /vehicle/command after RSS + occ + jerk."""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node

from autocar_command_gate.gate import Cmd, CommandGate, select_lead, slice_to_xy
from autocar_msgs.msg import (
    BrainHealth,
    GateStatus,
    OccupancySlice,
    SupervisorState,
    TrackedObjects,
    VehicleCommand,
    VehicleState,
)

_ST = {0: "INIT", 1: "ARMING", 2: "RUNNING", 3: "DEGRADED", 4: "HOLD", 5: "ESTOP", 6: "DISENGAGED"}
_VERDICT = {"pass": 0, "clip": 1, "hold": 2}


class GateNode(Node):
    def __init__(self) -> None:
        super().__init__("gate_node")
        self._gate = CommandGate()
        self._sup = "INIT"
        self._incoming: Cmd | None = None
        self._occ = np.zeros((0, 2), dtype=np.float32)
        self._pose = (0.0, 0.0, 0.0)
        self._v = 0.0
        self._collision = False
        self._objects: list[tuple[int, float, float]] = []
        self._pub = self.create_publisher(VehicleCommand, "/vehicle/command", 10)
        self._status = self.create_publisher(GateStatus, "/gate/status", 10)
        self._health = self.create_publisher(BrainHealth, "/gate/health", 10)
        self.create_subscription(SupervisorState, "/supervisor/state", self._on_sup, 10)
        self.create_subscription(VehicleCommand, "/brains/a/command", self._on_cmd, 10)
        self.create_subscription(VehicleCommand, "/brains/b/command", self._on_cmd, 10)
        self.create_subscription(VehicleCommand, "/brains/c/command", self._on_cmd, 10)
        self.create_subscription(OccupancySlice, "/perception/occupancy_slice", self._on_occ, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/localization/pose", self._on_pose, 10)
        self.create_subscription(VehicleState, "/vehicle/state", self._on_state, 10)
        self.create_subscription(TrackedObjects, "/perception/tracked_objects", self._on_obj, 10)
        self.create_timer(0.02, self._tick)

    def _on_sup(self, msg: SupervisorState) -> None:
        self._sup = _ST.get(int(msg.state), "INIT")

    def _on_cmd(self, msg: VehicleCommand) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._incoming = Cmd(
            stamp, msg.steering_tire_angle_rad, msg.accel_mps2, bool(msg.hold), int(msg.seq), msg.source
        )

    def _on_occ(self, msg: OccupancySlice) -> None:
        self._occ = slice_to_xy(msg)

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math_yaw(q.x, q.y, q.z, q.w)
        self._pose = (p.x, p.y, yaw)

    def _on_state(self, msg: VehicleState) -> None:
        self._v = float(msg.v_mps)
        self._collision = bool(msg.collision)

    def _on_obj(self, msg: TrackedObjects) -> None:
        self._objects = [
            (int(o.classification), o.pose.position.x, o.pose.position.y) for o in msg.objects
        ]

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        lead = select_lead(self._objects, self._pose, self._v, self._gate.p)
        out = self._gate.tick(now, self._incoming, self._sup, lead, self._occ, self._pose, self._v, self._collision)
        cmd = VehicleCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.source = out.cmd.source
        cmd.steering_tire_angle_rad = float(out.cmd.steer)
        cmd.accel_mps2 = float(out.cmd.accel)
        cmd.hold = bool(out.cmd.hold)
        cmd.seq = int(out.cmd.seq)
        self._pub.publish(cmd)
        st = GateStatus()
        st.header.stamp = cmd.header.stamp
        st.verdict = _VERDICT.get(out.verdict, 2)
        st.reason = out.reason
        st.v_target = self._v
        self._status.publish(st)
        h = BrainHealth()
        h.header.stamp = cmd.header.stamp
        h.brain = "gate"
        h.ok = True
        h.degraded = out.reason == "degraded"
        h.last_ok_stamp = now
        h.last_ms = 20.0
        self._health.publish(h)


def math_yaw(x: float, y: float, z: float, w: float) -> float:
    import math

    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> None:
    rclpy.init()
    node = GateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
