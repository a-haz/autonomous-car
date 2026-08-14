/* cage/autocar_mcu_bridge/firmware/heartbeat.h */
#pragma once
#include <stdint.h>

#define AC_SYNC0 0xA5u
#define AC_SYNC1 0xC3u
#define AC_TYPE_HOST_HB 0x01u
#define AC_TYPE_MCU_ACK 0x02u
#define AC_TYPE_ESTOP   0x03u
#define AC_MISS_LIMIT   3
#define AC_PERIOD_MS    20

typedef struct __attribute__((packed)) {
  uint8_t sync0, sync1, type, seq;
  uint32_t t_ms;
  uint8_t flags, brain_id, reserved, csum;
} ac_frame_t;

static inline uint8_t ac_csum(const ac_frame_t *f) {
  const uint8_t *p = (const uint8_t *)f;
  uint8_t c = 0;
  for (int i = 2; i <= 10; ++i) c ^= p[i];
  return c;
}

/* main loop (50 Hz timer): if (++miss >= AC_MISS_LIMIT) open_inhibit(); */
