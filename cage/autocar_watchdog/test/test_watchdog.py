"""second_miss is owned by the watchdog, per episode."""
from __future__ import annotations

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
