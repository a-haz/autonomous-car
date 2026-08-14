"""12-byte UART frame. Matches cage/autocar_mcu_bridge/firmware/heartbeat.h."""
from __future__ import annotations

import struct

SYNC0 = 0xA5
SYNC1 = 0xC3
TYPE_HOST_HB = 0x01
TYPE_MCU_ACK = 0x02
TYPE_ESTOP = 0x03
TYPE_ARM = 0x04
TYPE_DISARM = 0x05
TYPE_STATUS = 0x06
MISS_LIMIT = 3
PERIOD_S = 0.020
FRAME_LEN = 12

# HOST_HB flags
HB_ARMED = 1 << 0
HB_HOLD = 1 << 1
HB_ESTOP_REQUEST = 1 << 2

# MCU_ACK flags
ACK_INHIBIT_OPEN = 1 << 0
ACK_LATCH = 1 << 1
ACK_HW_ESTOP = 1 << 2


def csum_bytes(body: bytes) -> int:
    c = 0
    for b in body:
        c ^= b
    return c & 0xFF


def pack(typ: int, seq: int, t_ms: int, flags: int, brain_id: int) -> bytes:
    body = struct.pack("<BBIBBB", typ & 0xFF, seq & 0xFF, t_ms & 0xFFFFFFFF, flags & 0xFF, brain_id & 0xFF, 0)
    return bytes([SYNC0, SYNC1]) + body + bytes([csum_bytes(body)])


def unpack(frame: bytes) -> tuple[int, int, int, int, int]:
    if len(frame) != FRAME_LEN:
        raise ValueError("frame length")
    if frame[0] != SYNC0 or frame[1] != SYNC1:
        raise ValueError("sync")
    body = frame[2:11]
    if csum_bytes(body) != frame[11]:
        raise ValueError("csum")
    typ, seq, t_ms, flags, brain_id, _reserved = struct.unpack("<BBIBBB", body)
    return typ, seq, t_ms, flags, brain_id
