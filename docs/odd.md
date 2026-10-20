# Operational Design Domain (v0)

The supervisor evaluates this checklist. Items marked operator bit have no
sensor behind them; an operator sets them.

`odd.public_road_jp` is a signed file on disk, not a launch argument. It
defaults to false. See `deploy/autocar_bringup/config/odd_bits.yaml`.

## Checklist

| # | Condition | v0 allowed | If false |
| --- | --- | --- | --- |
| 1 | Geofence polygon (Lanelet site) | inside, 2 m hysteresis | HOLD |
| 2 | Map match: loc on a lanelet | yes | DEGRADED → HOLD |
| 3 | Time of day | daylight | HOLD. Operator bit (no sun sensor) |
| 4 | Precip / standing water | dry or light rain | HOLD. Operator bit |
| 5 | Speed | ≤ 40 km/h sim/campus; ≤ 8 m/s 1/10 | clamp then HOLD |
| 6 | Road type | mapped lanes with Lanelet | HOLD |
| 7 | Unprotected left across high-speed traffic | no until v1 | HOLD if planner requests it |
| 8 | Public-road Japan | no without written lab authorization + safety driver | refuse arm (signed file) |
| 9 | Safety driver | present until disengagement budget met | refuse unsupervised arm. Operator bit |
| 10 | Sensor health | lidar + (wide camera or GT perc) + IMU + MCU | DEGRADED / ESTOP |
| 11 | Loc covariance | `sqrt(P_xx+P_yy)` vs 0.8 m sim / 0.4 m 1/10 | DEGRADED |
| 12 | Surface | paved / parking lot | HOLD off-road. Operator bit |

## Speed and scene

- SIL / campus twin: 0–40 km/h, daylight, dry or light rain, mapped lanelets.
- 1/10: 0–8 m/s, parking lot / closed course.
- Procedural figure-8 is the G0–G4 scene. Campus / urban USD is decided after G4.

## Disengagement budget (written)

| Stage | Budget to drop the *requirement* for a safety driver |
| --- | --- |
| SIL G4 | 0 collisions, 0 red-light violations, 0 loc-lost-without-HOLD on the locked catalog × 3 seeds |
| 1/10 | 50 km closed-course, RC safety driver, < 1 disengage / 2 km, 0 contact. Hands-off is a lab decision, not a software flag. |
| Full-size | 500 km geofenced, driver in seat, number TBD with the lab. Software will not hide the driver. |

This is a research stack toward SAE L4 inside a written geofence. It is
not a claim that the car is L4.
