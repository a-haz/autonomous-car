"""User-space MCU. 3 missed HOST_HB (60 ms) → latch. No software clear."""
from __future__ import annotations

from autocar_mcu_bridge.frame import (
    ACK_HW_ESTOP,
    ACK_INHIBIT_OPEN,
    ACK_LATCH,
    MISS_LIMIT,
    PERIOD_S,
    TYPE_ESTOP,
    TYPE_HOST_HB,
    TYPE_MCU_ACK,
    pack,
    unpack,
)


class LoopbackMcu:
    def __init__(self) -> None:
        self.latched = False
        self.hw_estop = False
        self.miss = 0
        self.last_hb_s: float | None = None
        self.seq = 0

    def hardware_reset(self) -> None:
        """Keyed reset / power cycle. Not a ROS service."""
        self.latched = False
        self.hw_estop = False
        self.miss = 0
        self.last_hb_s = None

    def on_host_frame(self, now_s: float, frame: bytes) -> bytes | None:
        typ, _seq, _t_ms, flags, _brain = unpack(frame)
        if typ == TYPE_ESTOP or (flags & 0x04):
            self.latched = True
            self.hw_estop = True
        if typ == TYPE_HOST_HB:
            self.last_hb_s = now_s
            self.miss = 0
        return self.ack(now_s)

    def tick(self, now_s: float) -> bytes:
        if self.last_hb_s is None or (now_s - self.last_hb_s) >= PERIOD_S:
            self.miss += 1
            if self.miss >= MISS_LIMIT:
                self.latched = True
        return self.ack(now_s)

    def ack(self, now_s: float) -> bytes:
        flags = 0
        if self.latched:
            flags |= ACK_INHIBIT_OPEN | ACK_LATCH
        if self.hw_estop:
            flags |= ACK_HW_ESTOP
        self.seq = (self.seq + 1) & 0xFF
        return pack(TYPE_MCU_ACK, self.seq, int(now_s * 1000.0) & 0xFFFFFFFF, flags, 0)

    @property
    def mcu_ok(self) -> bool:
        return not self.latched
