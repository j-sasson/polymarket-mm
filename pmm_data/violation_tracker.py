"""Tracks constraint violations as episodes over time: when a violation first
appears, how its magnitude moves while it persists across subsequent price
updates, and when it resolves (stops appearing in the checker's output).

Decoupled from any transport: `update()` just takes a price snapshot and a
timestamp, so this can be driven by a live websocket, a CSV replay, or a
hand-built test sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pmm_data.constraints import ConstraintSet


@dataclass
class OpenEpisode:
    start_time: float
    constraint_name: str
    constraint_type: str
    market_ids: tuple[str, ...]
    magnitudes: list[float] = field(default_factory=list)


class ViolationTracker:
    """`on_episode_closed(record: dict)` fires once per resolved (or, on
    `flush()`, censored) violation episode."""

    def __init__(self, constraint_set: ConstraintSet, on_episode_closed: Callable[[dict], None]):
        self.constraint_set = constraint_set
        self.on_episode_closed = on_episode_closed
        self._open: dict[tuple, OpenEpisode] = {}

    def update(self, prices: dict[str, float], now_ts: float) -> None:
        violations = self.constraint_set.check(prices)
        current = {(v.constraint_name, v.market_ids): v for v in violations}

        for key, v in current.items():
            episode = self._open.get(key)
            if episode is None:
                self._open[key] = OpenEpisode(
                    start_time=now_ts,
                    constraint_name=v.constraint_name,
                    constraint_type=v.constraint_type,
                    market_ids=v.market_ids,
                    magnitudes=[v.magnitude],
                )
            else:
                episode.magnitudes.append(v.magnitude)

        for key in list(self._open):
            if key not in current:
                episode = self._open.pop(key)
                self._emit(episode, end_time=now_ts, resolved=True)

    def export_state(self) -> dict:
        """Serializable snapshot of currently-open episodes, for a caller
        that needs to persist across process restarts (e.g. one poll per
        invocation instead of one long-running process)."""
        return {
            "|".join((ep.constraint_name, *ep.market_ids)): {
                "start_time": ep.start_time,
                "constraint_name": ep.constraint_name,
                "constraint_type": ep.constraint_type,
                "market_ids": list(ep.market_ids),
                "magnitudes": list(ep.magnitudes),
            }
            for ep in self._open.values()
        }

    def load_state(self, state: dict) -> None:
        """Restores open episodes from `export_state`'s output. Replaces
        whatever open episodes this tracker currently has."""
        self._open = {}
        for entry in state.values():
            key = (entry["constraint_name"], tuple(entry["market_ids"]))
            self._open[key] = OpenEpisode(
                start_time=entry["start_time"],
                constraint_name=entry["constraint_name"],
                constraint_type=entry["constraint_type"],
                market_ids=tuple(entry["market_ids"]),
                magnitudes=list(entry["magnitudes"]),
            )

    def flush(self, now_ts: float) -> None:
        """Close out any still-open episodes as censored (process stopped
        while the violation was ongoing -- not a true convergence)."""
        for key in list(self._open):
            episode = self._open.pop(key)
            self._emit(episode, end_time=now_ts, resolved=False)

    def _emit(self, episode: OpenEpisode, end_time: float, resolved: bool) -> None:
        self.on_episode_closed({
            "constraint_name": episode.constraint_name,
            "constraint_type": episode.constraint_type,
            "market_ids": episode.market_ids,
            "start_time": episode.start_time,
            "end_time": end_time,
            "duration_seconds": round(end_time - episode.start_time, 3),
            "num_observations": len(episode.magnitudes),
            "start_magnitude": episode.magnitudes[0],
            "end_magnitude": episode.magnitudes[-1],
            "max_magnitude": max(episode.magnitudes),
            "mean_magnitude": sum(episode.magnitudes) / len(episode.magnitudes),
            "resolved": resolved,
        })
