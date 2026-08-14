"""Latency budgets. Owns Health.second_miss per episode_id."""
from __future__ import annotations


class Watchdog:
    def __init__(self) -> None:
        self.episode_id = 0
        self._first_miss = False
        self.second_miss = False

    def on_episode(self, episode_id: int) -> None:
        if episode_id != self.episode_id:
            self.episode_id = episode_id
            self._first_miss = False
            self.second_miss = False

    def tick(
        self,
        *,
        episode_id: int,
        state: str,
        loc_ok: bool,
        perc_ok: bool,
        brain_ok: bool,
        gate_ok: bool,
    ) -> bool:
        self.on_episode(episode_id)
        miss = not (loc_ok and perc_ok and brain_ok and gate_ok)
        if state == "RUNNING" and miss:
            self._first_miss = True
        elif state == "DEGRADED" and miss and self._first_miss:
            self.second_miss = True
        elif state == "RUNNING" and not miss:
            self._first_miss = False
        return self.second_miss
