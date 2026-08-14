# deploy/autocar_bringup/launch/vehicle.launch.py
"""Always-on cage + loc + occ + one shared Stanley. Exactly one brain overlay."""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from autocar_bringup.launch_contract import command_topics, validate_brain, validate_perc_source

_PKG_SHARE = Path(__file__).resolve().parents[1]
_CYCLONE = _PKG_SHARE / "cyclone_localhost.xml"


def _setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    brain = validate_brain(LaunchConfiguration("active_brain").perform(context))
    perc = validate_perc_source(LaunchConfiguration("perc_source").perform(context))

    common = {"use_sim_time": str(use_sim_time).lower()}
    traj_topic, cmd_topic = command_topics(brain)

    nodes = [
        Node(
            package="autocar_supervisor",
            executable="supervisor_node",
            parameters=[common, {"active_brain": brain}],
        ),
        Node(package="autocar_command_gate", executable="gate_node", parameters=[common]),
        Node(package="autocar_watchdog", executable="watchdog_node", parameters=[common]),
        Node(
            package="autocar_occ",
            executable="occ_node",
            parameters=[common, {"backend": "lidar" if perc == "gt" else "nvblox"}],
        ),
        Node(package="autocar_loc", executable="loc_node", parameters=[common]),
        Node(
            package="autocar_mcu_bridge",
            executable="mcu_node",
            parameters=[common, {"backend": "loopback" if use_sim_time else "uart"}],
        ),
        Node(
            package="autocar_vehicle_interface",
            executable="iface_node",
            parameters=[common, {"backend": "isaac" if use_sim_time else "vesc"}],
        ),
        Node(
            package="autocar_ctrl",
            executable="stanley_pid_node",
            name="shared_ctrl",
            remappings=[
                ("~/trajectory", traj_topic),
                ("~/command", cmd_topic),
            ],
            parameters=[common],
        ),
    ]
    if perc == "gt":
        nodes.append(
            Node(package="autocar_perc_gt", executable="gt_node", parameters=[common])
        )
    else:
        nodes.append(
            Node(package="autocar_perc", executable="perc_node", parameters=[common])
        )
    # exactly one brain overlay — switch = relaunch, not lifecycle hot-swap
    if brain == "A":
        nodes.append(
            Node(
                package="autocar_plan",
                executable="plan_node",
                parameters=[common, {"mode": "centerline"}],
            )
        )
    elif brain == "B":
        nodes.append(
            Node(
                package="autocar_autoware_adapter",
                executable="adapter_node",
                parameters=[common, {"mode": "traj"}],
            )
        )
    elif brain == "C":
        nodes.append(
            Node(
                package="autocar_alpamayo_adapter",
                executable="adapter_node",
                parameters=[common, {"backend": "stub"}],
            )
        )
    return nodes


def generate_launch_description() -> LaunchDescription:
    cyclone_uri = f"file://{_CYCLONE}"
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value="rmw_cyclonedds_cpp"),
            SetEnvironmentVariable(name="CYCLONEDDS_URI", value=cyclone_uri),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("active_brain", default_value="A"),
            DeclareLaunchArgument("perc_source", default_value="gt"),
            OpaqueFunction(function=_setup),
        ]
    )
