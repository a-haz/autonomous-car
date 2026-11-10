"""PR-02 MCU frame + 60 ms latch."""
from __future__ import annotations

import re
from pathlib import Path

from autocar_mcu_bridge import frame
from autocar_mcu_bridge.frame import (
    ACK_LATCH,
    FRAME_LEN,
    MISS_LIMIT,
    PERIOD_S,
    TYPE_HOST_HB,
    TYPE_MCU_ACK,
    pack,
    unpack,
)
from autocar_mcu_bridge.loopback import LoopbackMcu


def test_roundtrip_and_csum() -> None:
    raw = pack(TYPE_HOST_HB, 7, 1234, 0x01, 1)
    assert len(raw) == FRAME_LEN
    typ, seq, t_ms, flags, brain = unpack(raw)
    assert (typ, seq, t_ms, flags, brain) == (TYPE_HOST_HB, 7, 1234, 0x01, 1)


def test_bad_csum_rejected() -> None:
    raw = bytearray(pack(TYPE_HOST_HB, 0, 0, 0, 0))
    raw[11] ^= 0xFF
    try:
        unpack(bytes(raw))
    except ValueError as e:
        assert "csum" in str(e)
    else:
        raise AssertionError("expected csum error")


def test_three_misses_latch_and_no_software_clear() -> None:
    m = LoopbackMcu()
    now = 0.0
    m.on_host_frame(now, pack(TYPE_HOST_HB, 0, 0, 0, 0))
    assert not m.latched
    for i in range(MISS_LIMIT):
        now += PERIOD_S
        ack = m.tick(now)
        _typ, _s, _t, flags, _b = unpack(ack)
        if i + 1 >= MISS_LIMIT:
            assert m.latched
            assert flags & ACK_LATCH
    m.on_host_frame(now + PERIOD_S, pack(TYPE_HOST_HB, 1, 0, 0, 0))
    assert m.latched
    assert not m.mcu_ok
    m.hardware_reset()
    assert not m.latched
    assert m.mcu_ok


def test_latch_at_60_ms_for_any_clock_epoch() -> None:
    for t0 in (0.0, 1000.0, 123456.789, 1.7e9):
        m = LoopbackMcu()
        m.on_host_frame(t0, pack(TYPE_HOST_HB, 0, 0, 0, 0))
        for k in range(1, MISS_LIMIT + 1):
            m.tick(t0 + k * PERIOD_S)
            assert m.latched is (k == MISS_LIMIT), (t0, k)


def test_heartbeat_from_before_a_clock_rewind_does_not_count() -> None:
    m = LoopbackMcu()
    m.on_host_frame(1000.0, pack(TYPE_HOST_HB, 0, 0, 0, 0))
    for k in range(1, MISS_LIMIT + 1):
        m.tick(k * PERIOD_S)
    assert m.latched


def test_heartbeat_from_before_a_clock_rewind_stays_stale_when_the_clock_climbs_back() -> None:
    m = LoopbackMcu()
    m.on_host_frame(10.0, pack(TYPE_HOST_HB, 0, 0, 0, 0))
    for t in (9.98, 10.0, 10.02):
        m.tick(t)
    assert m.latched


def test_heartbeat_each_period_never_latches() -> None:
    m = LoopbackMcu()
    t = 1000.0
    for _ in range(500):
        m.on_host_frame(t, pack(TYPE_HOST_HB, 0, 0, 0, 0))
        m.tick(t + 1e-6)
        t += PERIOD_S
    assert not m.latched


def test_ack_type() -> None:
    m = LoopbackMcu()
    ack = m.on_host_frame(0.0, pack(TYPE_HOST_HB, 0, 0, 0, 0))
    assert unpack(ack)[0] == TYPE_MCU_ACK


def test_firmware_header_matches_frame_constants() -> None:
    header = (Path(__file__).resolve().parents[1] / "firmware" / "heartbeat.h").read_text()
    defines = {
        name: int(value, 0)
        for name, value in re.findall(r"#define AC_(\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", header)
    }
    names = [n for n in dir(frame) if n.startswith(("SYNC", "TYPE_", "HB_", "ACK_")) or n == "MISS_LIMIT"]
    assert len(names) == 15
    for name in names:
        assert defines[name] == getattr(frame, name), name
    assert defines["PERIOD_MS"] == round(frame.PERIOD_S * 1000)
