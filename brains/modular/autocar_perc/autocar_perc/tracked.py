"""Tracker ABI. Pure Python, no ROS.

One typed object list for every perception source, mirroring
messages/autocar_msgs/msg/TrackedObject.idl and TrackedObjects.idl.
The planner reads the TrackedObjects-shaped dict from to_msg() and cannot
tell which source produced it (docs/design.md, Perception).
"""
from __future__ import annotations

from dataclasses import dataclass, field

TRACKED_OBJECTS_TOPIC = "/perception/tracked_objects"
RATE_HZ = 10.0
FRAME_ID = "map"

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

CLASSES = range(CLASS_UNKNOWN, CLASS_STATIC + 1)
LIGHTS = range(LIGHT_UNKNOWN, LIGHT_OFF + 1)
OBJECT_KEYS = ("id", "classification", "classification_score", "light_state", "pose",
               "dimensions", "twist", "yaw_rate", "existence_prob", "age_frames")


def _vec(v, n: int, name: str) -> tuple[float, ...]:
    t = tuple(float(a) for a in v)
    if len(t) != n:
        raise ValueError(f"{name} must have {n} values, got {len(t)}")
    return t


@dataclass(slots=True)
class TrackedObject:
    id: int
    classification: int
    pose: tuple[float, float, float, float]  # x, y, z, yaw in the map frame
    dimensions: tuple[float, float, float]  # l, w, h
    twist: tuple[float, float, float]  # vx, vy, vz
    classification_score: float = 1.0
    light_state: int = LIGHT_UNKNOWN
    yaw_rate: float = 0.0
    existence_prob: float = 1.0
    age_frames: int = 1

    def __post_init__(self) -> None:
        if self.id < 0 or self.id > 0xFFFFFFFF:
            raise ValueError("id must fit uint32")
        if self.classification not in CLASSES:
            raise ValueError(f"unknown classification {self.classification}")
        if self.light_state not in LIGHTS:
            raise ValueError(f"unknown light_state {self.light_state}")
        if not 0 <= self.age_frames <= 255:
            raise ValueError("age_frames must fit uint8")
        self.pose = _vec(self.pose, 4, "pose")
        self.dimensions = _vec(self.dimensions, 3, "dimensions")
        self.twist = _vec(self.twist, 3, "twist")
        if min(self.dimensions) < 0.0:
            raise ValueError("dimensions must be >= 0")

    @property
    def xy(self) -> tuple[float, float]:
        return (self.pose[0], self.pose[1])

    def to_msg(self) -> dict:
        return {
            "id": self.id,
            "classification": self.classification,
            "classification_score": self.classification_score,
            "light_state": self.light_state,
            "pose": self.pose,
            "dimensions": self.dimensions,
            "twist": self.twist,
            "yaw_rate": self.yaw_rate,
            "existence_prob": self.existence_prob,
            "age_frames": self.age_frames,
        }

    @classmethod
    def from_msg(cls, m: dict) -> TrackedObject:
        missing = [k for k in OBJECT_KEYS if k not in m]
        if missing:
            raise ValueError(f"TrackedObject missing {missing}")
        return cls(id=int(m["id"]), classification=int(m["classification"]), pose=m["pose"],
                   dimensions=m["dimensions"], twist=m["twist"],
                   classification_score=float(m["classification_score"]),
                   light_state=int(m["light_state"]), yaw_rate=float(m["yaw_rate"]),
                   existence_prob=float(m["existence_prob"]), age_frames=int(m["age_frames"]))


@dataclass(slots=True)
class TrackedObjects:
    stamp_ns: int
    objects: list[TrackedObject] = field(default_factory=list)
    frame_id: str = FRAME_ID

    def to_msg(self) -> dict:
        """TrackedObjects-shaped dict for TRACKED_OBJECTS_TOPIC."""
        return {"stamp_ns": self.stamp_ns, "frame_id": self.frame_id,
                "objects": [o.to_msg() for o in self.objects]}

    @classmethod
    def from_msg(cls, m: dict) -> TrackedObjects:
        return cls(int(m["stamp_ns"]), [TrackedObject.from_msg(o) for o in m["objects"]],
                   str(m["frame_id"]))
