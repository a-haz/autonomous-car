"""g0-unit: PR-01 files and frozen IDL names exist. No ROS required."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MSGS = ROOT / "messages" / "autocar_msgs"


def test_frozen_idl_present() -> None:
    for name in (
        "TrackedObject.idl",
        "TrackedObjects.idl",
        "Trajectory.idl",
        "TrajectoryPoint.idl",
        "VehicleCommand.idl",
        "VehicleState.idl",
        "SupervisorState.idl",
    ):
        assert (MSGS / "msg" / name).is_file(), name


def test_companion_idl_present() -> None:
    for name in (
        "BrainHealth.idl",
        "GateStatus.idl",
        "McuFrame.idl",
        "RadarDetection.idl",
        "RadarDetections.idl",
        "LaneletMapMeta.idl",
        "OccupancySlice.idl",
        "PredictedObject.idl",
        "PredictedObjects.idl",
        "TrafficLightState.idl",
        "TrafficLightStates.idl",
    ):
        assert (MSGS / "msg" / name).is_file(), name


def test_inject_services_present() -> None:
    for name in ("InjectCutIn.srv", "InjectRedLight.srv", "InjectLocDrop.srv", "Arm.srv"):
        assert (MSGS / "srv" / name).is_file(), name


def test_trajectory_bound_is_64() -> None:
    text = (MSGS / "msg" / "Trajectory.idl").read_text()
    assert "TRAJ_N = 64" in text
    assert "sequence<autocar_msgs::msg::TrajectoryPoint, 64>" in text


def test_predicted_object_uses_trajectory_point() -> None:
    text = (MSGS / "msg" / "PredictedObject.idl").read_text()
    assert '#include "autocar_msgs/msg/TrajectoryPoint.idl"' in text
    assert "sequence<autocar_msgs::msg::TrajectoryPoint, 64>" in text


def test_traffic_light_duplicates_constants() -> None:
    text = (MSGS / "msg" / "TrafficLightState.idl").read_text()
    assert "LIGHT_RED = 1" in text
    assert "TrackedObject" not in text


def test_cmake_depends_std_msgs() -> None:
    text = (MSGS / "CMakeLists.txt").read_text()
    assert "std_msgs" in text
    assert "geometry_msgs" in text


def test_docs_extracts_exist() -> None:
    for name in ("odd.md", "safety.md", "bom.md", "design.md"):
        assert (ROOT / "docs" / name).is_file(), name


def test_firmware_header_sync_bytes() -> None:
    header = (
        ROOT / "cage" / "autocar_mcu_bridge" / "firmware" / "heartbeat.h"
    ).read_text()
    assert "0xA5" in header
    assert "0xC3" in header
    assert "AC_MISS_LIMIT   3" in header


def test_no_ira_warehouse_imports_in_isaac_code() -> None:
    isaac = ROOT / "simulation" / "isaac"
    if not isaac.exists():
        return
    code = {".py", ".usda", ".usd", ".usdc"}
    for path in isaac.rglob("*"):
        if path.suffix.lower() not in code:
            continue
        text = path.read_text(errors="ignore")
        assert "import ira_warehouse" not in text
        assert "ira_warehouse_8" not in text
        assert "occlusion_v2" not in text
        assert "omni.isaac.sensor" not in text
        assert "omni.isaac.dynamic_control" not in text
