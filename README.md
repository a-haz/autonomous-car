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

**PR-01** (messages + bring-up skeleton) is in tree. Next is **PR-02**
(supervisor, command gate, MCU loopback). Do not start with Brain C or
1/10 hardware.

- Design: [docs/design.md](docs/design.md)
- ODD / safety / BOM: [docs/odd.md](docs/odd.md), [docs/safety.md](docs/safety.md), [docs/bom.md](docs/bom.md)

```bash
# Host contract tests (no ROS required)
python3 -m pytest -q

# IDL + launch package (needs ROS 2 Jazzy)
source /opt/ros/jazzy/setup.bash
colcon build --packages-select autocar_msgs autocar_bringup
source install/setup.bash
ros2 launch autocar_bringup vehicle.launch.py use_sim_time:=true active_brain:=A perc_source:=gt
```

The launch file is the contract. Overlay nodes are skeletons until later
PRs; bringing the graph up now will fail on missing executables. That is
expected.

## Machine notes

- SIL host: Isaac Sim **6.0.1** (`~/isaacsim`), RTX 5070 Ti. GUI-async
  only. Isaac 5.1.0 is parked on this GPU.
- Headless `SimulationApp` is unsafe on this Blackwell card.
- 1/10 compute: Jetson Orin NX, JetPack 6.2 / Humble / Isaac ROS 3.2.

## License

MIT. See [LICENSE](LICENSE).
