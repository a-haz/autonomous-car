# Autonomous car

Research SAE L4 stack for a geofenced campus / private-road car.

Isaac Sim first, then a 1/10-scale Ackermann vehicle. Three autonomy
brains share one vehicle interface and one safety cage:

| Brain | What it is |
| --- | --- |
| A | From-scratch modular stack (perception → plan → control) |
| B | Autoware Core / Universe, adapted to our command messages |
| C | NVIDIA Alpamayo 1 Nano (trajectory-out VLA) |

Only one brain is armed. A command gate (RSS + occupancy hold) and an
independent watchdog MCU sit under every brain, including the neural one.

This is **not** the Tanaka-lab mega-rover thesis. Public-road type
approval, ISO 26262 certification, and a robotaxi fleet are out of scope.

## Status

Design is accepted (2026-08-14). See [docs/design.md](docs/design.md).

Implementation starts at **PR-01** (frozen messages + bring-up skeleton)
and follows: cage → Brain A in sim → Autoware adapter → Alpamayo
contract → 1/10 Brain A. Do not start with Brain C or 1/10 hardware.

## Machine notes

- Sim host: Isaac Sim 5.1.0, RTX 5070 Ti. GUI-async only (headless
  `SimulationApp` is unsafe on this Blackwell card).
- 1/10 compute target: Jetson Orin NX (Thor if Brain C must run 10B on-car).

## License

MIT. See [LICENSE](LICENSE).
