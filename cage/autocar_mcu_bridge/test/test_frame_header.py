"""Check frame.py against firmware/heartbeat.h so the two cannot drift."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from autocar_mcu_bridge import frame
from autocar_mcu_bridge.frame import FRAME_LEN, SYNC0, SYNC1, TYPE_HOST_HB, csum_bytes, pack, unpack

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "heartbeat.h"
_DEFINE = re.compile(r"^#define\s+AC_(\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\s*$", re.MULTILINE)


def _header_defines() -> dict[str, int]:
    return {name: int(value, 0) for name, value in _DEFINE.findall(HEADER.read_text())}


def test_header_defines_match_python_constants() -> None:
    defines = _header_defines()
    assert defines, "no AC_ defines parsed from heartbeat.h"
    for name, value in defines.items():
        if name == "PERIOD_MS":
            assert round(frame.PERIOD_S * 1000) == value
            continue
        assert hasattr(frame, name), f"frame.py has no constant for AC_{name}"
        assert getattr(frame, name) == value, f"AC_{name}"


def test_csum_covers_bytes_2_to_10_like_firmware() -> None:
    raw = pack(TYPE_HOST_HB, 0x5A, 0x01020304, 0x07, 0x09)
    expected = 0
    for i in range(2, 11):
        expected ^= raw[i]
    assert raw[11] == expected == csum_bytes(raw[2:11])


def test_field_offsets_match_packed_struct() -> None:
    raw = pack(TYPE_HOST_HB, 0x5A, 0x01020304, 0x07, 0x09)
    assert len(raw) == FRAME_LEN == 12
    assert raw[0] == SYNC0 and raw[1] == SYNC1
    assert raw[2] == TYPE_HOST_HB
    assert raw[3] == 0x5A
    assert raw[4:8] == bytes([0x04, 0x03, 0x02, 0x01])
    assert raw[8] == 0x07
    assert raw[9] == 0x09
    assert raw[10] == 0


@pytest.mark.parametrize("raw", [b"", bytes(FRAME_LEN - 1), bytes(FRAME_LEN + 1)])
def test_unpack_rejects_wrong_length(raw: bytes) -> None:
    with pytest.raises(ValueError, match="frame length"):
        unpack(raw)


@pytest.mark.parametrize("index", [0, 1])
def test_unpack_rejects_bad_sync(index: int) -> None:
    raw = bytearray(pack(TYPE_HOST_HB, 0, 0, 0, 0))
    raw[index] ^= 0xFF
    with pytest.raises(ValueError, match="sync"):
        unpack(bytes(raw))
