"""G1 GT perception: ground-truth object poses to TrackedObjects. Pure Python.

The output has the same shape as the trt perception path, so the planner
cannot tell which source is running (perc_source:=gt|trt).
Class and light values mirror messages/autocar_msgs/msg/TrackedObject.idl.
"""
from __future__ import annotations

from dataclasses import dataclass

CLASS_UNKNOWN = 0
CLASS_CAR = 1
CLASS_TRUCK = 2
CLASS_PED = 3
CLASS_CYCLIST = 4
CLASS_TRAFFIC_LIGHT = 5
CLASS_STATIC = 6

LIGHT_UNKNOWN = 0
LIGHT_RED = 1
LIGHT_YELLOW = 2
LIGHT_GREEN = 3
LIGHT_OFF = 4


@dataclass(slots=True)
class GtObj:
    id: int
    cls: int
    x: float
    y: float
    yaw: float
    l: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    light: int = LIGHT_UNKNOWN


def to_tracked(stamp_ns: int, objects: list[GtObj]) -> dict:
    """GT poses in map frame to a TrackedObjects-shaped dict."""
    return {
        "stamp_ns": stamp_ns,
        "frame_id": "map",
        "objects": [
            {
                "id": o.id,
                "classification": o.cls,
                "classification_score": 1.0,
                "light_state": o.light,
                "pose": (o.x, o.y, 0.0, o.yaw),
                "dimensions": (o.l, o.w, o.h),
                "twist": (o.vx, o.vy, 0.0),
                "yaw_rate": 0.0,
                "existence_prob": 1.0,
                "age_frames": 1,
            }
            for o in objects
        ],
    }
