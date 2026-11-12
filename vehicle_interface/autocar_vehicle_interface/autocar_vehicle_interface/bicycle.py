"""Bicycle twin backend for vehicle_interface (PR-03a). Pure Python, no ROS.

This is the design's backend:=bicycle. It wraps autocar_ctrl.bicycle.step
so open-loop and closed-loop tests can run without Isaac.
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
