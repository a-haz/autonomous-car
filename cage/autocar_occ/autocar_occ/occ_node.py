"""OccupancySlice from lidar 2D projection. Inflate once."""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
import struct

from sensor_msgs.msg import PointCloud2

from autocar_msgs.msg import OccupancySlice
from autocar_occ.slice import from_lidar_xy


def _xyz_xy(msg: PointCloud2) -> list[tuple[float, float]]:
    names = [f.name for f in msg.fields]
    if not {"x", "y", "z"} <= set(names):
        return []
    off = {f.name: f.offset for f in msg.fields}
    pts: list[tuple[float, float]] = []
    step = msg.point_step
    data = bytes(msg.data)
    for i in range(0, len(data), step):
        x = struct.unpack_from("<f", data, i + off["x"])[0]
        y = struct.unpack_from("<f", data, i + off["y"])[0]
        z = struct.unpack_from("<f", data, i + off["z"])[0]
        pts.append((x, y, z))
    return pts


class OccNode(Node):
    def __init__(self) -> None:
        super().__init__("occ_node")
        self.declare_parameter("backend", "lidar")
        self.declare_parameter("r_inflate", 0.30)
        self.declare_parameter("track", 1.55)
        self.declare_parameter("bumper_z", 0.40)
        self._pub = self.create_publisher(OccupancySlice, "/perception/occupancy_slice", 10)
        self.create_subscription(PointCloud2, "/sensors/lidar/points", self._on_cloud, 10)

    def _on_cloud(self, msg: PointCloud2) -> None:
        bumper = float(self.get_parameter("bumper_z").value)
        pts = [(x, y) for x, y, z in _xyz_xy(msg) if abs(float(z) - bumper) < 0.50]
        xy = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        sl = from_lidar_xy(
            xy,
            r_inflate=float(self.get_parameter("r_inflate").value),
            track=float(self.get_parameter("track").value),
            bumper_z=bumper,
        )
        out = OccupancySlice()
        out.header = msg.header
        out.header.frame_id = "map"
        out.resolution = sl.resolution
        out.width = sl.width
        out.height = sl.height
        out.origin_x = sl.origin_x
        out.origin_y = sl.origin_y
        out.bumper_z = sl.bumper_z
        out.inflate_m = sl.inflate_m
        out.cells = sl.cells.flatten().tolist()
        out.source = "lidar"
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = OccNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
