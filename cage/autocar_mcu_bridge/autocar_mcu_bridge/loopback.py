"""User-space MCU. 3 missed HOST_HB (60 ms) → latch. No software clear."""
from __future__ import annotations

import math

from autocar_mcu_bridge.frame import (
    ACK_HW_ESTOP,
    ACK_INHIBIT_OPEN,
    ACK_LATCH,
    HB_ESTOP_REQUEST,
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
        self._last_now_s = -math.inf

    def _observe(self, now_s: float) -> None:
        # A heartbeat from before a backward clock jump never counts again,
        # however far the clock climbs back.
        if now_s < self._last_now_s:
            self.last_hb_s = None
        self._last_now_s = now_s

    def hardware_reset(self) -> None:
        """Keyed reset / power cycle. Not a ROS service."""
        self.latched = False
        self.hw_estop = False
        self.miss = 0
        self.last_hb_s = None

    def on_host_frame(self, now_s: float, frame: bytes) -> bytes | None:
        self._observe(now_s)
        typ, _seq, _t_ms, flags, _brain = unpack(frame)
        if typ == TYPE_ESTOP or (flags & HB_ESTOP_REQUEST):
            self.latched = True
            self.hw_estop = True
        if typ == TYPE_HOST_HB:
            self.last_hb_s = now_s
            self.miss = 0
        return self.ack(now_s)

    def tick(self, now_s: float) -> bytes:
        self._observe(now_s)
        # 1 us slack: (t0 + 0.02) - t0 rounds below 0.02 for most t0 != 0, which
        # skipped the first miss and latched at 80 ms instead of 60 ms.
        age = None if self.last_hb_s is None else now_s - self.last_hb_s
        if age is None or not 0.0 <= age < PERIOD_S - 1e-6:
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
