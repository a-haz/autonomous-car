cmake_minimum_required(VERSION 3.8)
project(autocar_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# One struct per .idl file. ROS 2 rosidl treats two structs as a service pair.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedObject.idl"
  "msg/TrackedObjects.idl"
  "msg/TrajectoryPoint.idl"
  "msg/Trajectory.idl"
  "msg/VehicleCommand.idl"
  "msg/VehicleState.idl"
  "msg/SupervisorState.idl"
  "msg/BrainHealth.idl"
  "msg/GateStatus.idl"
  "msg/McuFrame.idl"
  "msg/RadarDetection.idl"
  "msg/RadarDetections.idl"
  "msg/LaneletMapMeta.idl"
  "msg/OccupancySlice.idl"
  "msg/PredictedObject.idl"
  "msg/PredictedObjects.idl"
  "msg/TrafficLightState.idl"
  "msg/TrafficLightStates.idl"
  "srv/InjectCutIn.srv"
  "srv/InjectRedLight.srv"
  "srv/InjectLocDrop.srv"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
