# Autonomous car

A research SAE L4 stack for a geofenced campus or private-road car. It runs
in Isaac Sim first, then on a 1/10-scale Ackermann vehicle.

Three autonomy brains share one vehicle interface and one safety cage:

| Brain | What it is |
| --- | --- |
| A | Modular stack written from scratch (perception, planning, control) |
| B | Autoware Core / Universe, adapted to our command messages |
| C | NVIDIA Alpamayo 1 Nano (trajectory-out VLA) |

Only one brain is armed at a time. A command gate (RSS plus occupancy hold)
and an independent watchdog MCU sit under every brain, including the neural
one.

This is not the Tanaka-lab mega-rover thesis. Public-road type approval,
ISO 26262 certification and a robotaxi fleet are out of scope.

## Status

PR-02 is in the tree: supervisor, command gate, MCU loopback and occupancy.
PR-03a (bicycle-in-the-loop G0, no Isaac) is next. Brain C and the 1/10
hardware come later.

The design is in [docs/design.md](docs/design.md). The ODD, safety notes and
bill of materials are in [docs/odd.md](docs/odd.md),
[docs/safety.md](docs/safety.md) and [docs/bom.md](docs/bom.md).

## Build and test

```bash
# Host contract tests (no ROS required)
python3 -m pytest -q

# IDL and launch package (needs ROS 2 Jazzy)
source /opt/ros/jazzy/setup.bash
colcon build --packages-select autocar_msgs autocar_bringup
source install/setup.bash
ros2 launch autocar_bringup vehicle.launch.py use_sim_time:=true active_brain:=A perc_source:=gt
```

The launch file is the contract. Most overlay nodes are still skeletons, so
bringing the full graph up fails on missing executables for now.

## Machine notes

The SIL host runs Isaac Sim 6.0.1 (`~/isaacsim`) on an RTX 5070 Ti, GUI-async
only. Isaac 5.1.0 is parked on this GPU, and headless `SimulationApp` crashes
on this Blackwell card. The 1/10 car uses a Jetson Orin NX with JetPack 6.2,
ROS 2 Humble and Isaac ROS 3.2.

## License

MIT. See [LICENSE](LICENSE).
