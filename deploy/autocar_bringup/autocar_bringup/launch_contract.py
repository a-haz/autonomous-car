"""Launch argument contract. Kept importable without ROS so g0-unit can run on a bare host."""

from __future__ import annotations

BRAINS = ("A", "B", "C", "NONE")
PERC_SOURCES = ("gt", "trt")

ALWAYS_ON = (
    "autocar_supervisor",
    "autocar_command_gate",
    "autocar_watchdog",
    "autocar_occ",
    "autocar_loc",
    "autocar_mcu_bridge",
    "autocar_vehicle_interface",
    "autocar_ctrl",
)

OVERLAY = {
    "A": "autocar_plan",
    "B": "autocar_autoware_adapter",
    "C": "autocar_alpamayo_adapter",
    "NONE": None,
}


def validate_brain(raw: str) -> str:
    brain = raw.strip().upper()
    if brain not in BRAINS:
        raise ValueError(f"active_brain must be A|B|C|NONE, got {raw!r}")
    return brain


def validate_perc_source(raw: str) -> str:
    perc = raw.strip().lower()
    if perc not in PERC_SOURCES:
        raise ValueError("perc_source must be gt|trt")
    return perc


def overlay_package(brain: str) -> str | None:
    return OVERLAY[validate_brain(brain)]


def command_topics(brain: str) -> tuple[str, str]:
    """Shared Stanley remaps. Brains publish Trajectory only."""
    key = validate_brain(brain)
    slug = "none" if key == "NONE" else key.lower()
    return (f"/brains/{slug}/trajectory", f"/brains/{slug}/command")
