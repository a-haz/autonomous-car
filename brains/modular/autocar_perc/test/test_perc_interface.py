"""PR-07a perception interface tests: tracker ABI and perc_source flag. No ROS."""
from __future__ import annotations

import re
from pathlib import Path

import autocar_perc.tracked as abi
import numpy as np
import pytest
from autocar_bringup.launch_contract import PERC_SOURCES as LAUNCH_PERC_SOURCES
from autocar_perc.source import (
    PERC_SOURCES,
    GtSource,
    PercSourceUnavailable,
    Perception,
    TrtSource,
    make_backend,
    validate_perc_source,
)
from autocar_perc.tracked import (
    CLASS_CAR,
    CLASS_STATIC,
    OBJECT_KEYS,
    TRACKED_OBJECTS_TOPIC,
    TrackedObject,
    TrackedObjects,
)
from autocar_perc_gt.stub import GtObj, to_tracked
from autocar_plan.simple_planner import CenterlinePlanner

ROOT = Path(__file__).resolve().parents[4]
MSG = ROOT / "messages" / "autocar_msgs" / "msg"
TOPICS = ROOT / "deploy" / "autocar_bringup" / "config" / "topics.yaml"


def _objs() -> list[GtObj]:
    return [GtObj(7, CLASS_STATIC, 1.0, 2.0, 0.5, 4.0, 1.8, 1.5),
            GtObj(8, CLASS_CAR, -3.0, 0.0, 0.0, 4.5, 1.9, 1.6, vx=2.0)]


# ABI

def test_constants_match_idl() -> None:
    found = dict(re.findall(r"\bconst \w+ (\w+) = (\d+);", (MSG / "TrackedObject.idl").read_text()))
    assert found
    for name, value in found.items():
        assert getattr(abi, name) == int(value), name


def test_object_fields_match_idl() -> None:
    body = (MSG / "TrackedObject.idl").read_text().split("struct TrackedObject {")[1].split("};")[0]
    fields = re.findall(r"(\w+);", body)
    assert tuple(fields) == OBJECT_KEYS
    assert set(TrackedObject.__slots__) == set(OBJECT_KEYS)
    assert "sequence<autocar_msgs::msg::TrackedObject> objects" in (MSG / "TrackedObjects.idl").read_text()


def test_topic_matches_bringup() -> None:
    assert f"tracked_objects: {TRACKED_OBJECTS_TOPIC}" in TOPICS.read_text()


def test_abi_carries_id_class_pose_size_velocity_stamp() -> None:
    t = TrackedObjects(42, [TrackedObject(3, CLASS_CAR, (1, 2, 0, 0.5), (4.5, 1.9, 1.6), (2.0, 0.5, 0.0))])
    m = t.to_msg()
    assert m["stamp_ns"] == 42 and m["frame_id"] == "map"
    (o,) = m["objects"]
    assert set(o) == set(OBJECT_KEYS)
    assert o["id"] == 3 and o["classification"] == CLASS_CAR
    assert o["pose"] == (1.0, 2.0, 0.0, 0.5)
    assert o["dimensions"] == (4.5, 1.9, 1.6)
    assert o["twist"] == (2.0, 0.5, 0.0)
    assert TrackedObjects.from_msg(m) == t


@pytest.mark.parametrize("kw,match", [
    ({"classification": 9}, "classification"),
    ({"light_state": 7}, "light_state"),
    ({"id": -1}, "uint32"),
    ({"age_frames": 300}, "uint8"),
    ({"pose": (0.0, 0.0, 0.0)}, "pose"),
    ({"dimensions": (1.0, -1.0, 1.0)}, "dimensions"),
])
def test_abi_rejects_bad_fields(kw: dict, match: str) -> None:
    base = {"id": 1, "classification": CLASS_CAR, "pose": (0, 0, 0, 0), "dimensions": (1, 1, 1),
            "twist": (0, 0, 0)}
    with pytest.raises(ValueError, match=match):
        TrackedObject(**(base | kw))


def test_from_msg_rejects_missing_keys() -> None:
    m = to_tracked(0, _objs())["objects"][0]
    del m["twist"]
    with pytest.raises(ValueError, match="twist"):
        TrackedObject.from_msg(m)


# perc_source flag

def test_flag_values_match_launch_contract() -> None:
    assert PERC_SOURCES == LAUNCH_PERC_SOURCES


def test_flag_selects_gt() -> None:
    assert validate_perc_source(" GT ") == "gt"
    assert isinstance(make_backend("gt", _objs), GtSource)
    p = Perception("gt", _objs)
    assert p.perc_source == "gt" and isinstance(p.backend, GtSource)


@pytest.mark.parametrize("bad", ["", "both", "lidar", "gt,trt"])
def test_flag_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError, match="perc_source must be gt|trt"):
        Perception(bad, _objs)


def test_gt_needs_object_source() -> None:
    with pytest.raises(ValueError, match="ground-truth"):
        Perception("gt")


def test_trt_fails_clearly_and_never_detects() -> None:
    for build in (lambda: Perception("trt", _objs), lambda: make_backend("TRT"), lambda: TrtSource()):
        with pytest.raises(PercSourceUnavailable, match="not available in this build"):
            build()
    stub = object.__new__(TrtSource)
    with pytest.raises(PercSourceUnavailable, match="not available in this build"):
        stub.detect(0)


def test_msg_before_step_fails() -> None:
    with pytest.raises(RuntimeError, match="step"):
        Perception("gt", _objs).msg()


# Contract: the planner cannot tell the interface from the stub

def test_interface_msg_equals_stub() -> None:
    p = Perception("gt", _objs)
    t = p.step(123)
    assert isinstance(t, TrackedObjects) and all(isinstance(o, TrackedObject) for o in t.objects)
    direct, via = to_tracked(123, _objs()), p.msg()
    assert type(via) is type(direct)
    assert via == direct
    for a, b in zip(via["objects"], direct["objects"]):
        assert list(a) == list(b)
        assert all(type(a[k]) is type(b[k]) for k in a)


def test_planner_output_identical_via_interface() -> None:
    s = np.linspace(0.0, 60.0, 241)
    lane = np.column_stack([s, np.zeros_like(s)])
    objs = [GtObj(1, CLASS_STATIC, 20.0, 0.1, 0.0, 1.0, 1.0, 1.0),
            GtObj(2, CLASS_STATIC, 30.0, 4.0, 0.0, 1.0, 1.0, 1.0)]
    lights = {"lights": []}
    p = Perception("gt", lambda: objs)
    for pose in ((0.0, 0.0, 0.0), (5.0, 0.2, 0.05), (12.0, -0.1, -0.02)):
        a = CenterlinePlanner(lane, 1, 5.0).step(pose, to_tracked(0, objs), lights)
        p.step(0)
        b = CenterlinePlanner(lane, 1, 5.0).step(pose, p.msg(), lights)
        assert type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b)
