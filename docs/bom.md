# Bill of materials

Student / research street prices, mid-2026, USD. Bands, not quotes.
**Phase 2 is not purchased.** Full-size donor is chosen at quote time ~2027
(Dataspeed-class DBW only; no garage CAN).

## Phase 0: already owned

| Item | PN / loc | Notes |
| --- | --- | --- |
| Workstation `max` | RTX 5070 Ti 16 GB, Ubuntu 24.04 | SIL host |
| Isaac Sim 6.0.1 | `~/isaacsim` | **SIL lock**, GUI-async only |
| Isaac Sim 5.1.0 | `~/isaacsim-5.1.0` | **parked** on this 5070 Ti |
| PC73 overflow | RTX 5090 32 GB | lab; official Alpamayo 10B, not the nightly SIL host |

## Phase 1: 1/10, buy for G5

| Item | Representative PN | Qty | Power | Cost band (USD) | Notes |
| --- | --- | --- | --- | --- | --- |
| 1/10 4WD Ackermann | Traxxas Slash 4×4 VXL 68086-4 | 1 | drivetrain | 350–500 | F1TENTH-class |
| Motor ESC | VESC 6 MkV or Flipsky 75/300 | 1 | 5–15 ctrl | 150–280 | ODrive as alt |
| Compute | Jetson Orin NX 16 GB `900-13767-0000` + carrier | 1 | 15–25 (40 max) | 700–1 200 | JP6.2 / Humble / Isaac ROS 3.2 |
| Storage | WD SN580 1 TB NVMe | 1 | 3 | 60–90 | |
| Lidar (default) | Livox Mid-360 | 1 | 6.5 | 700–900 | 100BASE-TX, **no PoE** |
| Lidar (optional 2D) | Hokuyo UST-10LX | 0–1 | 2.4 | 1 400–1 800 | not v0 perc |
| Camera | Intel RealSense D455 `82635D455` | 1 | 3.5 | 380–450 | |
| Safety MCU | NUCLEO-G474RE (STM32G474) | 1 | <1 | 15–25 | |
| RC override | Flysky FS-i6S + iA6B or FrSky | 1 | 0.2 | 40–70 | independent of Jetson |
| E-stop | 16 mm NC mushroom + 30 A relay | 1 | - | 20–40 | series with MCU inhibit |
| GNSS (lot) | u-blox ZED-F9P breakout | 1 | 0.3 | 180–250 | optional |
| IMU extra | ICM-42688-P breakout | 1 | 0.1 | 15–25 | |
| Packs | 4S 8000 mAh drivetrain + **4S or 12 V** compute | 2 | - | 80–160 | **never 6S into Orin** |
| Mid-360 Ethernet | Cat5e, **no PoE** | 1 | - | 10–20 | 1 GbE port on carrier |
| PDB / UBEC / wiring | - | 1 | - | 80–150 | |
| Prints, bumpers, mast | PETG / TPU | 1 | - | 50–120 | |
| **v0 buy (no Hokuyo)** | | | **~30–40 W sense+compute** | **~$2 800–4 300** | |

No radar on 1/10. Brain C is not armed on 1/10 v0.

## Phase 2: full-size, not purchased

| Item | Class | Cost band (USD) | Status |
| --- | --- | --- | --- |
| Donor | Dataspeed-supported list **at quote time (~2027)** | 8 k–25 k used | not purchased |
| DBW kit | Dataspeed ADAS / By-Wire | 40 k–80 k | **no garage CAN** |
| Roof lidar | Ouster OS1-64 / OS2-128 | 8 k–18 k | |
| Radars | Continental / Aptiv corner ×4 | 4 k–12 k | |
| Cameras | 8× GMSL2 + deserializer | 4 k–12 k | Holoscan later |
| GNSS/INS | NovAtel / Applanix class | 8 k–25 k | RTK |
| Compute | x86 + 5070-class or AGX Thor | 4 k–12 k | |
| Safety MCU + brake interrupt | STM32G4 + hardware relay | 200–500 | not ASIL-D |
| Dual power + UPS | isolated 12/24 V, DC-UPS 300 W | 1 k–3 k | |
| Cabin e-stop, rack | - | 1 k–3 k | |
| **Rollup ex-donor** | | **~$70 k–160 k** | do not buy before G5 |
| **With donor** | | **~$80 k–185 k** | |

No new body. No EV powertrain project. No “ASIL-D the Jetson.”

Source: [design.md](design.md) § BOM.
