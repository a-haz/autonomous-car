# autocar_msgs

Frozen cage types plus the companion messages specified in
`docs/design.md`.

ROS 2 rosidl allows **one struct per `.idl` file** (two structs are parsed
as a service). Nested types therefore live in their own files
(`TrajectoryPoint`, `TrackedObjects`, `VehicleState`, …) and are
referenced with fully-qualified names:

```
sequence<autocar_msgs::msg::TrajectoryPoint, 64> points;
```

That is the same ABI as the design’s combined listings.
