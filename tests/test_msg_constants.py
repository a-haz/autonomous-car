"""IDL constants: the cage code agrees with them, and rosidl generates them."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from autocar_command_gate.gate import LEAD_CLASSES
from autocar_supervisor.sm import Brain, St

MSG_DIR = Path(__file__).resolve().parents[1] / "messages" / "autocar_msgs" / "msg"


def _constants() -> dict[str, dict[str, int]]:
    out = {}
    for path in sorted(MSG_DIR.glob("*.idl")):
        text = path.read_text()
        found = re.findall(r"\bconst \w+ (\w+) = (\d+);", text)
        if found:
            out[path.stem] = {name: int(value) for name, value in found}
    return out


def _prefixed(consts: dict[str, int], prefix: str) -> dict[str, int]:
    return {k.removeprefix(prefix): v for k, v in consts.items() if k.startswith(prefix)}


def test_cage_code_matches_idl_constants(ros) -> None:
    c = _constants()
    assert _prefixed(c["SupervisorState"], "ST_") == {s.name: s.value for s in St}
    assert _prefixed(c["SupervisorState"], "BRAIN_") == {b.name: b.value for b in Brain}
    leads = ("CLASS_CAR", "CLASS_TRUCK", "CLASS_PED", "CLASS_CYCLIST")
    assert {c["TrackedObject"][k] for k in leads} == LEAD_CLASSES
    assert (c["Trajectory"]["TRAJ_N"], c["VehicleCommand"]["GEAR_DRIVE"]) == (64, 4)
    gate_node = importlib.import_module("autocar_command_gate.gate_node")
    watchdog_node = importlib.import_module("autocar_watchdog.watchdog_node")
    assert {k.upper(): v for k, v in gate_node._VERDICT.items()} == _prefixed(c["GateStatus"], "VERDICT_")
    for states in (gate_node._ST, watchdog_node._ST):
        assert {name: value for value, name in states.items()} == _prefixed(c["SupervisorState"], "ST_")


def test_generated_messages_carry_idl_constants() -> None:
    """Needs the colcon overlay sourced; skipped by the host-only run."""
    msg = pytest.importorskip("autocar_msgs.msg", reason="autocar_msgs not built and sourced")
    for struct, consts in _constants().items():
        cls = getattr(msg, struct)
        for name, value in consts.items():
            assert getattr(cls, name, None) == value, (struct, name)
