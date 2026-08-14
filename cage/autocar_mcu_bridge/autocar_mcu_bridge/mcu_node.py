"""MCU host + loopback backend. 50 Hz heartbeat, 60 ms latch."""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from autocar_mcu_bridge.frame import TYPE_HOST_HB, pack, unpack
from autocar_mcu_bridge.loopback import LoopbackMcu
from autocar_msgs.msg import BrainHealth, McuFrame


class McuNode(Node):
    def __init__(self) -> None:
        super().__init__("mcu_node")
        self.declare_parameter("backend", "loopback")
        self._mcu = LoopbackMcu()
        self._seq = 0
        self._pub_rx = self.create_publisher(McuFrame, "/mcu/rx", 10)
        self._pub_tx = self.create_publisher(McuFrame, "/mcu/tx", 10)
        self._health = self.create_publisher(BrainHealth, "/mcu/health", 10)
        self.create_timer(0.02, self._tick)

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self._seq = (self._seq + 1) & 0xFF
        hb = pack(TYPE_HOST_HB, self._seq, int(now * 1000.0) & 0xFFFFFFFF, 0, 0)
        tx = McuFrame()
        tx.header.stamp = self.get_clock().now().to_msg()
        tx.type, tx.seq, tx.t_ms, tx.flags, tx.brain_id = unpack(hb)
        self._pub_tx.publish(tx)
        ack = self._mcu.on_host_frame(now, hb)
        self._mcu.tick(now + 1e-6)
        if ack:
            rx = McuFrame()
            rx.header.stamp = tx.header.stamp
            rx.type, rx.seq, rx.t_ms, rx.flags, rx.brain_id = unpack(ack)
            self._pub_rx.publish(rx)
        h = BrainHealth()
        h.header.stamp = tx.header.stamp
        h.brain = "mcu"
        h.ok = self._mcu.mcu_ok
        h.degraded = self._mcu.latched
        h.note = "latched" if self._mcu.latched else ""
        self._health.publish(h)


def main() -> None:
    rclpy.init()
    node = McuNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
