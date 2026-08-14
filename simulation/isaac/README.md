# Isaac SIL

Host: `max`, Isaac Sim **6.0.1** at `~/isaacsim`. GUI-async only.
Headless `SimulationApp` segfaults on this Blackwell card.

Do **not**:

- import `omni.isaac.sensor` or `omni.isaac.dynamic_control` (removed in 6.0)
- import anything from `ira_warehouse_8` or `occlusion_v2`
- add the IRA warehouse / NavMesh pedestrian scene

Use `isaacsim.sensors.rtx` and the current `isaacsim.ros2.bridge`. Verify
extension names on the box in PR-03b.
