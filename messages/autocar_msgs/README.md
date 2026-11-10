# autocar_msgs

Frozen cage types plus the companion messages specified in
`docs/design.md`.

ROS 2 rosidl allows one struct per `.idl` file (two structs are parsed
as a service). Nested types therefore live in their own files
(`TrajectoryPoint`, `TrackedObjects`, `VehicleState` and so on) and are
referenced with fully-qualified names:

```
sequence<autocar_msgs::msg::TrajectoryPoint, 64> points;
```

Constants go in a `module <Struct>_Constants { ... };` block next to the
struct (as `rosidl_adapter` generates them). rosidl silently drops
constants declared directly in `module msg`.

The ABI matches the types described in the design doc.
