# Safety

Three independent stop paths. Any one is sufficient.

1. **Software HOLD** — command gate / supervisor.
2. **MCU inhibit** — missed heartbeat or `ESTOP` frame.
3. **Hard e-stop + RC override** — wired, not a ROS topic.

A brain that publishes `hold=false` at 50 Hz cannot close a relay the MCU
opened. There is no ROS service that clears an ESTOP latch.

## Command gate

Every `/vehicle/command` passes the gate at **50 Hz**. Stale command (age
> 80 ms) → comfort HOLD at `a_hold = 3.0 m/s²`. RSS / occupancy / chassis
collision → emergency HOLD at `a_min_brake_rear`.

RSS v0 is **Rule 1 only**, with separate rear/front decelerations
(`a_min_brake_rear` vs `a_max_brake_front`). After a jerk clip the gate
re-checks RSS. We do not claim Rules 2–5.

`DEGRADED` is a gate actuator: last steer held, speed driven toward
`v_degrade` (4.0 m/s sim, 1.5 m/s 1/10) using `−a_hold`, then RSS/occ
may still escalate.

Verdict strings (ABI): `pass` | `clip` | `hold`.

## MCU

STM32G4 class (NUCLEO-G474RE). UART 115200 8N1, 12-byte frame, 50 Hz.

| | |
| --- | --- |
| Sync | `0xA5 0xC3` |
| Timeout | 3 missed host frames (**60 ms**) → FET/relay opens |
| Clear | hardware reset pin or power cycle. **No ROS clear.** |
| Firmware | in-tree `cage/autocar_mcu_bridge/firmware/heartbeat.h` |

Never common-ground the e-stop inhibit through the Jetson USB.

## 1/10 RC / e-stop wiring

```
RC Rx CH3 ──┐
            ├── hardware mux ── PWM ── VESC
Orin cmd  ──┘     ^
                  └── failsafe = RC if RSSI lost
E-stop NC mushroom ── series ── MCU inhibit FET ── VESC ENABLE
```

- RC override is **independent** of the Jetson.
- Mux failsafe is RC when RSSI is lost.
- Mushroom is NC, series with the MCU inhibit FET on VESC ENABLE.

Phase 2: cabin mushroom drops Dataspeed DBW enable. Same three-path idea.

## Legal

- Research platform. MIT code. **No** claim of roadworthiness.
- Japan public roads: forbidden unless a signed `public_road_jp` file is
  present. Not a launch argument.
- Campus / private road: safety driver until the written budget.
- Phase 2: Dataspeed + donor insurance + university rules. Not purchased.

## Privacy

Bags will contain faces and plates.

| Rule | v0 |
| --- | --- |
| Retention | 30 days on `max`; 7 days on the 1/10 NVMe |
| Public GitHub | no raw bags, no raw images |
| Publish | blur faces/plates before any paper figure |
| Third-party models | no campus-bag upload without a written exception |

## Threats (short)

| Threat | Mitigation |
| --- | --- |
| Spoofed GNSS | NDT must agree; > 2 m for 1 s → ignore GNSS, DEGRADED |
| Replayed sensors | MCU + e-stop do not trust perception |
| Compromised teleop | **No network teleop** on 1/10 v0; RC only |
| Weight swap | `BrainHealth.engine_sha256`; unsigned → refuse arm |
| Host “disables” MCU | physically impossible; sim-only flag compiled out of metal |

Source: [design.md](design.md) § Safety / Security.
