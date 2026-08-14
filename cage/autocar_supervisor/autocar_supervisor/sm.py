# cage/autocar_supervisor/autocar_supervisor/sm.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Brain(IntEnum):
    NONE = 0
    A = 1
    B = 2
    C = 3


class St(IntEnum):
    INIT = 0
    ARMING = 1
    RUNNING = 2
    DEGRADED = 3
    HOLD = 4
    ESTOP = 5
    DISENGAGED = 6


T_RECOVER_S = 1.0
DT = 0.02
BRAIN_DEATH_S = 0.300


@dataclass(slots=True)
class Health:
    odd: bool
    loc: bool
    perc: bool
    brain: bool
    mcu: bool
    gate: bool
    e_stop: bool
    collision: bool
    second_miss: bool
    cmd_age_s: float


class Supervisor:
    """Single-threaded SM. Matches the transition table line-for-line. 50 Hz."""

    def __init__(self) -> None:
        self.state = St.INIT
        self.brain = Brain.NONE
        self.reason = "boot"
        self.episode_id = 0
        self._want: Brain | None = None
        self._healthy_s = 0.0
        self._brain_bad_s = 0.0
        self._op: str | None = None

    def _enter_hold(self, reason: str) -> None:
        self.state = St.HOLD
        self.reason = reason
        self._want = None

    def request_arm(self, brain: Brain) -> None:
        if brain is Brain.NONE:
            return
        if self.state is St.INIT:
            self._want = brain
        elif self.state is St.HOLD:
            self._want = brain
            self._op = "arm"

    def request_hold(self) -> None:
        if self.state in (St.RUNNING, St.DEGRADED):
            self._op = "hold"

    def request_disengage(self) -> None:
        if self.state in (St.HOLD, St.RUNNING, St.DEGRADED):
            self._op = "disengage"

    def request_resume(self) -> None:
        if self.state is St.HOLD and self.brain is not Brain.NONE:
            self._want = self.brain
            self._op = "resume"

    def request_reset(self) -> None:
        if self.state is St.DISENGAGED:
            self._op = "reset"

    def step(self, h: Health) -> St:
        if h.e_stop or not h.mcu or h.collision:
            self.state = St.ESTOP
            self.reason = "collision" if h.collision else ("mcu" if not h.mcu else "estop")
            self.brain = Brain.NONE
            self._want = None
            self._op = None
            self._brain_bad_s = 0.0
            return self.state

        if h.brain:
            self._brain_bad_s = 0.0
        else:
            self._brain_bad_s += DT
        brain_dead = self._brain_bad_s >= BRAIN_DEATH_S

        if self.state is St.INIT:
            if self._want in (Brain.A, Brain.B, Brain.C) and h.odd and h.loc and h.mcu and h.gate:
                self.brain = self._want
                self.state = St.ARMING
                self.reason = "arming"
            elif self._want in (Brain.A, Brain.B, Brain.C):
                self.reason = "arm_precond"
        elif self.state is St.ARMING:
            if not (h.odd and h.loc and h.mcu and h.gate):
                self.state = St.INIT
                self.brain = Brain.NONE
                self.reason = "arm_precond"
            elif h.brain and h.odd and h.cmd_age_s < 0.080:
                self.state = St.RUNNING
                self.reason = f"armed_{self.brain.name}"
                self._healthy_s = 0.0
        elif self.state is St.RUNNING:
            if self._op == "disengage":
                self.state = St.DISENGAGED
                self.brain = Brain.NONE
                self.reason = "operator"
            elif self._op == "hold" or not h.odd or brain_dead:
                self._enter_hold(
                    "odd" if not h.odd else ("brain_death" if brain_dead else "operator")
                )
            elif not (h.loc and h.perc and h.brain and h.gate):
                self.state = St.DEGRADED
                self.reason = "budget"
                self._healthy_s = 0.0
        elif self.state is St.DEGRADED:
            if self._op == "disengage":
                self.state = St.DISENGAGED
                self.brain = Brain.NONE
                self.reason = "operator"
            elif self._op == "hold" or h.second_miss or not h.odd or brain_dead:
                self._enter_hold("second_miss" if h.second_miss else "degraded_hold")
            elif h.loc and h.perc and h.brain and h.gate and h.odd:
                self._healthy_s += DT
                if self._healthy_s >= T_RECOVER_S:
                    self.state = St.RUNNING
                    self.reason = "recovered"
            else:
                self._healthy_s = 0.0
        elif self.state is St.HOLD:
            if self._op == "disengage":
                self.state = St.DISENGAGED
                self.brain = Brain.NONE
                self.reason = "operator"
            elif self._op in {"resume", "arm"} and self._want in (Brain.A, Brain.B, Brain.C):
                self.brain = self._want
                self.state = St.ARMING
                self.reason = "resume" if self._op == "resume" else "switch"
        elif self.state is St.DISENGAGED:
            if self._op == "reset":
                self.episode_id += 1
                self._want = None
                self.state = St.INIT
                self.reason = "reset"
        self._op = None
        return self.state
