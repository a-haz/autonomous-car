"""Shared Stanley lateral + PID longitudinal controller (PR-04). Pure Python.

Sign convention (matches bicycle.py, positive steer turns left):
  e        cross-track error, positive when the path is to the LEFT of the
           front axle (the car must steer left to reach it).
  theta_e  path yaw minus vehicle yaw, wrapped to [-pi, pi).
  steer    wrap(theta_e) + atan2(k * e, v + v_soft), clamped to +-delta_max.
Defaults follow the StanleyPID sketch in docs/design.md (Brain A).
"""
from __future__ import annotations

import math

from autocar_ctrl.bicycle import A_MAX, A_MIN, DELTA_MAX, clamp, wrap


class StanleyPID:
    def __init__(self, L: float, k: float = 2.4, kp: float = 1.2, ki: float = 0.05, kd: float = 0.0,
                 delta_max: float = DELTA_MAX, a_max: float = A_MAX, a_min: float = A_MIN,
                 v_soft: float = 0.5) -> None:
        self.L = L
        self.k = k
        self.kp, self.ki, self.kd = kp, ki, kd
        self.delta_max = delta_max
        self.a_max, self.a_min = a_max, a_min
        self.v_soft = v_soft
        self._ie = 0.0
        self._pe = 0.0

    def lateral(self, e: float, theta_e: float, v: float) -> float:
        delta = wrap(theta_e) + math.atan2(self.k * e, v + self.v_soft)
        return clamp(delta, -self.delta_max, self.delta_max)

    def longitudinal(self, v: float, v_ref: float, dt: float) -> float:
        err = v_ref - v
        self._ie = clamp(self._ie + err * dt, -4.0, 4.0)
        de = (err - self._pe) / max(dt, 1e-3)
        self._pe = err
        a = self.kp * err + self.ki * self._ie + self.kd * de
        return clamp(a, self.a_min, self.a_max)
