"""second_miss is owned by the watchdog, per episode."""
from __future__ import annotations

from autocar_supervisor.sm import Brain, Health, Supervisor
from autocar_watchdog.watchdog import Watchdog


def test_first_running_miss_is_not_second() -> None:
    w = Watchdog()
    assert w.tick(episode_id=0, state="RUNNING", loc_ok=False, perc_ok=True, brain_ok=True, gate_ok=True) is False


def test_second_miss_while_degraded() -> None:
    w = Watchdog()
    w.tick(episode_id=3, state="RUNNING", loc_ok=False, perc_ok=True, brain_ok=True, gate_ok=True)
    assert w.tick(episode_id=3, state="DEGRADED", loc_ok=False, perc_ok=True, brain_ok=True, gate_ok=True) is True


def test_episode_clears() -> None:
    w = Watchdog()
    w.tick(episode_id=1, state="RUNNING", loc_ok=False, perc_ok=True, brain_ok=True, gate_ok=True)
    w.tick(episode_id=1, state="DEGRADED", loc_ok=False, perc_ok=True, brain_ok=True, gate_ok=True)
    assert w.second_miss
    w.tick(episode_id=2, state="RUNNING", loc_ok=True, perc_ok=True, brain_ok=True, gate_ok=True)
    assert w.second_miss is False


def _lockstep(loc_ok_per_tick: list[bool]) -> list[tuple[str, str]]:
    """Supervisor and watchdog wired as the nodes are: the watchdog sees what the
    supervisor published (state after the step, flags it stepped on)."""
    sup, wd, second = Supervisor(), Watchdog(), False

    def health(loc: bool) -> Health:
        return Health(odd=True, loc=loc, perc=True, brain=True, mcu=True, gate=True, e_stop=False,
                      collision=False, second_miss=second, cmd_age_s=0.02)

    sup.request_arm(Brain.A)
    sup.step(health(True))
    sup.step(health(True))
    trace = []
    for loc in loc_ok_per_tick:
        st = sup.step(health(loc))
        second = wd.tick(episode_id=sup.episode_id, state=st.name, loc_ok=loc, perc_ok=True,
                         brain_ok=True, gate_ok=True)
        trace.append((st.name, sup.reason))
    return trace


def test_persistent_miss_reaches_hold_through_the_supervisor() -> None:
    trace = _lockstep([True] + [False] * 10)
    assert trace[0][0] == "RUNNING"
    assert trace[1][0] == "DEGRADED"
    assert ("HOLD", "second_miss") in trace[:5]


def test_single_miss_recovers_without_hold() -> None:
    trace = _lockstep([True, False] + [True] * 60)
    assert trace[1][0] == "DEGRADED"
    assert all(state != "HOLD" for state, _ in trace)
    assert trace[-1][0] == "RUNNING"


def test_first_miss_published_as_degraded_is_not_second() -> None:
    w = Watchdog()
    assert w.tick(episode_id=0, state="DEGRADED", loc_ok=False, perc_ok=True, brain_ok=True, gate_ok=True) is False
    assert w.tick(episode_id=0, state="DEGRADED", loc_ok=False, perc_ok=True, brain_ok=True, gate_ok=True) is True
