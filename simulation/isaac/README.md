# Isaac SIL

The SIL host is `max` with Isaac Sim 6.0.1 at `~/isaacsim`. It runs GUI-async
only, because headless `SimulationApp` segfaults on this Blackwell card.

Isaac 6.0 removed `omni.isaac.sensor` and `omni.isaac.dynamic_control`, so the
SIL code uses `isaacsim.sensors.rtx` and the current `isaacsim.ros2.bridge`.
Exact extension names get checked on the box in PR-03b. Nothing here imports
from the thesis scenes (`ira_warehouse_8`, `occlusion_v2`) or uses the IRA
warehouse NavMesh pedestrians.
