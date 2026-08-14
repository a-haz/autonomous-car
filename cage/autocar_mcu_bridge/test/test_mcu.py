"""PR-02 MCU frame + 60 ms latch."""
from __future__ import annotations

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


def test_ack_type() -> None:
    m = LoopbackMcu()
    ack = m.on_host_frame(0.0, pack(TYPE_HOST_HB, 0, 0, 0, 0))
    assert unpack(ack)[0] == TYPE_MCU_ACK
