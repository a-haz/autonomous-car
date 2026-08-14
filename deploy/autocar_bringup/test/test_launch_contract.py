"""g0-unit: launch argument contract. No ROS graph required."""

from __future__ import annotations

import pytest

from autocar_bringup.launch_contract import (
    ALWAYS_ON,
    command_topics,
    overlay_package,
    validate_brain,
    validate_perc_source,
)


def test_brains_accepted() -> None:
    for raw in ("a", "B", "c", "none", "NONE"):
        validate_brain(raw)


def test_brain_rejected() -> None:
    with pytest.raises(ValueError, match="active_brain"):
        validate_brain("D")


def test_perc_accepted() -> None:
    assert validate_perc_source("GT") == "gt"
    assert validate_perc_source("trt") == "trt"


def test_perc_rejected() -> None:
    with pytest.raises(ValueError, match="perc_source"):
        validate_perc_source("both")


def test_stanley_remaps_per_brain() -> None:
    assert command_topics("A") == ("/brains/a/trajectory", "/brains/a/command")
    assert command_topics("C") == ("/brains/c/trajectory", "/brains/c/command")
    assert command_topics("NONE") == ("/brains/none/trajectory", "/brains/none/command")


def test_overlay_is_exactly_one() -> None:
    assert overlay_package("A") == "autocar_plan"
    assert overlay_package("B") == "autocar_autoware_adapter"
    assert overlay_package("C") == "autocar_alpamayo_adapter"
    assert overlay_package("NONE") is None


def test_cage_is_always_on() -> None:
    assert "autocar_supervisor" in ALWAYS_ON
    assert "autocar_command_gate" in ALWAYS_ON
    assert "autocar_ctrl" in ALWAYS_ON
    assert "autocar_loc" in ALWAYS_ON
    assert "autocar_occ" in ALWAYS_ON
