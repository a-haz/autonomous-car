"""PR-02 supervisor table tests. No ROS."""
from __future__ import annotations

from autocar_supervisor.sm import BRAIN_DEATH_S, DT, T_RECOVER_S, Brain, Health, St, Supervisor


def _ok(**kw) -> Health:
    base = dict(
        odd=True,
        loc=True,
        perc=True,
        brain=True,
        mcu=True,
        gate=True,
        e_stop=False,
        collision=False,
        second_miss=False,
        cmd_age_s=0.02,
    )
    base.update(kw)
    return Health(**base)


def _to_running(s: Supervisor) -> None:
    s.request_arm(Brain.A)
    assert s.step(_ok()) is St.ARMING
    assert s.step(_ok()) is St.RUNNING


def test_hold_persists_50_ticks_then_resume() -> None:
    s = Supervisor()
    _to_running(s)
    s.request_hold()
    assert s.step(_ok()) is St.HOLD
    assert s._want is None
    for _ in range(50):
        assert s.step(_ok()) is St.HOLD
    s.request_resume()
    assert s.step(_ok()) is St.ARMING


def test_leftover_want_does_not_auto_resume() -> None:
    s = Supervisor()
    s.request_arm(Brain.A)
    assert s._want is Brain.A
    assert s.step(_ok(odd=False, loc=False, gate=False)) is St.INIT
    _to_running(s)
    s.request_hold()
    s.step(_ok())
    s._want = Brain.A
    s._op = None
    assert s.step(_ok()) is St.HOLD


def test_arm_from_disengaged_is_ignored() -> None:
    s = Supervisor()
    _to_running(s)
    s.request_disengage()
    assert s.step(_ok()) is St.DISENGAGED
    s.request_arm(Brain.B)
    assert s.step(_ok()) is St.DISENGAGED
    s.request_reset()
    assert s.step(_ok()) is St.INIT
    assert s.episode_id == 1


def test_first_brain_miss_is_degraded() -> None:
    s = Supervisor()
    _to_running(s)
    assert s.step(_ok(brain=False)) is St.DEGRADED


def test_brain_death_hold() -> None:
    s = Supervisor()
    _to_running(s)
    ticks = int(BRAIN_DEATH_S / DT)
    st = s.step(_ok(brain=False))
    assert st is St.DEGRADED
    for _ in range(ticks):
        st = s.step(_ok(brain=False))
    assert st is St.HOLD


def test_collision_from_disengaged_is_estop() -> None:
    s = Supervisor()
    _to_running(s)
    s.request_disengage()
    s.step(_ok())
    assert s.step(_ok(collision=True)) is St.ESTOP
    s.request_reset()
    assert s.step(_ok()) is St.ESTOP


def test_recover_after_t_recover() -> None:
    s = Supervisor()
    _to_running(s)
    assert s.step(_ok(loc=False)) is St.DEGRADED
    n = int(T_RECOVER_S / DT)
    st = St.DEGRADED
    for _ in range(n):
        st = s.step(_ok())
    assert st is St.RUNNING


def test_second_miss_holds() -> None:
    s = Supervisor()
    _to_running(s)
    assert s.step(_ok(loc=False)) is St.DEGRADED
    assert s.step(_ok(loc=False, second_miss=True)) is St.HOLD
