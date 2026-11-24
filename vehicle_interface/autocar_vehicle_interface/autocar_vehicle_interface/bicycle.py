"""Bicycle twin backend for vehicle_interface (PR-03a). Pure Python, no ROS.

This is the design's backend:=bicycle. It wraps autocar_ctrl.bicycle.step
so open-loop and closed-loop tests can run without Isaac. An optional
simulated 2D lidar (lidar2d.SimLidar2D, PR-06b) can be attached; it
raycasts from the true pose and uses its own random stream.
"""
from __future__ import annotations

import math

from autocar_ctrl.bicycle import BicycleParams, BicycleState, clamp, step as plant_step


class BicycleBackend:
    def __init__(self, state: BicycleState | None = None, params: BicycleParams | None = None) -> None:
        self.params = params if params is not None else BicycleParams()
        self.state = state if state is not None else BicycleState()
        self.delta = 0.0
        self.accel = 0.0
        self.lidar = None  # opt-in SimLidar2D (lidar2d.py); None keeps the old sensor path

    def attach_lidar(self, lidar) -> None:
        self.lidar = lidar

    def lidar_scan(self):
        """One 2D scan raycast from the true pose. Requires attach_lidar()."""
        if self.lidar is None:
            raise RuntimeError("no lidar attached")
        s = self.state
        return self.lidar.scan(s.x, s.y, s.yaw)

    def set_command(self, delta: float, accel: float) -> None:
        self.delta = delta
        self.accel = accel

    def step(self, dt: float) -> BicycleState:
        self.state = plant_step(self.state, self.delta, self.accel, dt, self.params)
        return self.state

    @property
    def applied_delta(self) -> float:
        return clamp(self.delta, -self.params.delta_max, self.params.delta_max)

    @property
    def yaw_rate(self) -> float:
        return self.state.v * math.tan(self.applied_delta) / self.params.L
