"""Kinematic bicycle model (PR-04). Pure Python, no ROS.

Frame: x forward, y left, yaw counter-clockwise from +x. Positive steer
turns left. Limits mirror the twin vehicle params in docs/design.md
(Simulation section): L = 2.70 m, delta_max = 0.60 rad, v_max = 11.11 m/s.
A_MAX matches GateParams.a_max_accel in the command gate; A_MIN matches the
StanleyPID default a_min in the design.md Brain A sketch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

L_TWIN = 2.70
DELTA_MAX = 0.60
V_MAX = 11.11
A_MAX = 2.5
A_MIN = -3.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def bicycle_step(x: float, y: float, yaw: float, v: float, delta: float, dt: float, L: float) -> tuple[float, float, float]:
    """Exact-arc integration at constant v and delta over dt."""
    if abs(delta) < 1e-6:
        return x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw
    kappa = math.tan(delta) / L
    dpsi = v * kappa * dt
    r = 1.0 / kappa
    x2 = x + r * (math.sin(yaw + dpsi) - math.sin(yaw))
    y2 = y - r * (math.cos(yaw + dpsi) - math.cos(yaw))
    return x2, y2, wrap(yaw + dpsi)


@dataclass(slots=True)
class BicycleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0


@dataclass(frozen=True, slots=True)
class BicycleParams:
    L: float = L_TWIN
    delta_max: float = DELTA_MAX
    v_max: float = V_MAX
    a_max: float = A_MAX
    a_min: float = A_MIN


def step(s: BicycleState, delta: float, accel: float, dt: float, p: BicycleParams = BicycleParams()) -> BicycleState:
    """Clamp the command, move along the arc at the current v, then update v.

    Speed is clamped to [0, v_max]; this model does not reverse.
    """
    delta = clamp(delta, -p.delta_max, p.delta_max)
    accel = clamp(accel, p.a_min, p.a_max)
    x, y, yaw = bicycle_step(s.x, s.y, s.yaw, s.v, delta, dt, p.L)
    v = clamp(s.v + accel * dt, 0.0, p.v_max)
    return BicycleState(x, y, yaw, v)
