"""PR-05 GT perception stub tests. No ROS."""
from __future__ import annotations

import re
from pathlib import Path

import autocar_perc_gt.stub as stub
from autocar_perc_gt.stub import CLASS_CAR, CLASS_STATIC, GtObj, to_tracked

IDL = Path(__file__).resolve().parents[4] / "messages" / "autocar_msgs" / "msg" / "TrackedObject.idl"


def test_constants_match_idl() -> None:
    found = dict(re.findall(r"\bconst \w+ (\w+) = (\d+);", IDL.read_text()))
    assert found
    for name, value in found.items():
        assert getattr(stub, name) == int(value), name


def test_to_tracked_carries_pose_size_and_twist() -> None:
    out = to_tracked(123, [GtObj(7, CLASS_STATIC, 1.0, 2.0, 0.5, 4.0, 1.8, 1.5),
                           GtObj(8, CLASS_CAR, -3.0, 0.0, 0.0, 4.5, 1.9, 1.6, vx=2.0)])
    assert out["stamp_ns"] == 123 and out["frame_id"] == "map"
    a, b = out["objects"]
    assert a["id"] == 7 and a["classification"] == CLASS_STATIC
    assert a["pose"] == (1.0, 2.0, 0.0, 0.5)
    assert a["dimensions"] == (4.0, 1.8, 1.5)
    assert a["twist"] == (0.0, 0.0, 0.0)
    assert b["twist"] == (2.0, 0.0, 0.0)
    assert a["existence_prob"] == 1.0 and a["classification_score"] == 1.0


def test_tracked_has_no_gt_marker() -> None:
    """Same keys as a TrackedObject: nothing tells the planner it is GT."""
    keys = set(to_tracked(0, [GtObj(1, CLASS_CAR, 0, 0, 0, 1, 1, 1)])["objects"][0])
    assert keys == {"id", "classification", "classification_score", "light_state", "pose",
                    "dimensions", "twist", "yaw_rate", "existence_prob", "age_frames"}
