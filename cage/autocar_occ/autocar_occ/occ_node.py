"""OccupancySlice from lidar 2D projection. Inflate once."""
from __future__ import annotations

import numpy as np
import rclpy
from numpy.typing import NDArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from autocar_msgs.msg import OccupancySlice
from autocar_occ.slice import from_lidar_xy

_FLOAT_TYPES = {7: "f4", 8: "f8"}  # sensor_msgs/PointField FLOAT32, FLOAT64


def _xyz_xy(msg: PointCloud2) -> NDArray[np.float32]:
    """(N, 3) finite x, y, z of every point. Honours datatype, endianness and row padding."""
    fields = {f.name: f for f in msg.fields}
    if not {"x", "y", "z"} <= fields.keys():
        return np.zeros((0, 3), np.float32)
    order = ">" if msg.is_bigendian else "<"
    try:
        formats = [order + _FLOAT_TYPES[int(fields[n].datatype)] for n in "xyz"]
    except KeyError:
        raise ValueError("x/y/z must be FLOAT32 or FLOAT64") from None
    point = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": formats,
            "offsets": [int(fields[n].offset) for n in "xyz"],
            "itemsize": int(msg.point_step),
        }
    )
    height, width, row_step = int(msg.height), int(msg.width), int(msg.row_step)
    rows = np.frombuffer(bytes(msg.data), np.uint8, count=height * row_step).reshape(height, row_step)
    pts = np.ascontiguousarray(rows[:, : width * point.itemsize]).view(point).reshape(-1)
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


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
        try:
            xyz = _xyz_xy(msg)
        except ValueError as e:  # malformed cloud: say so rather than kill the node
            self.get_logger().error(f"dropping /sensors/lidar/points: {e}")
            return
        xy = xyz[np.abs(xyz[:, 2] - bumper) < 0.50, :2]
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
