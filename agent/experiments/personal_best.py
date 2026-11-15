"""A run's personal bests, kept in pb.json in the run's own folder:

    {"map_name": "A03-Race", "time_ms": 25440, "time_episode": 174,
     "distance_m": 1466.0, "distance_episode": 120}

Each run starts afresh, so its [PB] lines and pb.json describe that run alone.
"""

import json
import os
from pathlib import Path

FILE_NAME = "pb.json"


class PersonalBest:
    def __init__(self, run_dir, map_name: str):
        self.path = Path(run_dir) / FILE_NAME
        self.entry = {"map_name": map_name}

    @property
    def time_ms(self):
        return self.entry.get("time_ms") or float("inf")

    @property
    def distance_m(self):
        return self.entry.get("distance_m", 0.0)

    def update(self, episode, finished, race_time_ms, distance_m):
        """Record an episode. Returns ("time", previous ms) for a faster finish,
        ("distance", previous m) for a further unfinished run, or None."""
        change = None
        if finished and race_time_ms is not None and race_time_ms < self.time_ms:
            change = ("time", self.time_ms)
            self.entry.update(time_ms=race_time_ms, time_episode=episode)
        elif not finished and distance_m is not None and distance_m > self.distance_m:
            change = ("distance", self.distance_m)
        if distance_m is not None and distance_m > self.distance_m:
            self.entry.update(distance_m=round(float(distance_m), 3), distance_episode=episode)
        if change:
            self._save()
        return change

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(self.entry, indent=2) + "\n")
        tmp.replace(self.path)
