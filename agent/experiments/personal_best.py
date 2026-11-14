"""Personal bests per map, kept in results/pb.json beside the runs.

One entry per map, so training on A01 isn't compared with A03's record:

    {"A03-Race": {"time_ms": 25440, "time_run": "...", "time_episode": 174,
                  "distance_m": 1466.0, "distance_run": "...", "track_csv_sha256": "..."}}

The distance is only comparable on the same track data: when the map's track
CSV changes (e.g. the finish point is moved), the old distance is dropped and
the time, which the track data doesn't affect, is kept.
"""

import json
import os
from pathlib import Path

FILE_NAME = "pb.json"


class PersonalBest:
    def __init__(self, results_dir, map_name: str, track_csv_sha256: str = None):
        self.path = Path(results_dir) / FILE_NAME
        self.map_name = map_name
        self.track_csv_sha256 = track_csv_sha256
        entry = self._read().get(map_name, {})
        self.entry = dict(entry)
        if entry.get("track_csv_sha256") != track_csv_sha256:
            for key in ("distance_m", "distance_run"):
                self.entry.pop(key, None)
        self.entry["track_csv_sha256"] = track_csv_sha256

    @property
    def time_ms(self):
        return self.entry.get("time_ms") or float("inf")

    @property
    def distance_m(self):
        return self.entry.get("distance_m", 0.0)

    def update(self, run_id, episode, finished, race_time_ms, distance_m):
        """Record an episode. Returns ("time", previous ms) for a faster finish,
        ("distance", previous m) for a further unfinished run, or None."""
        change = None
        if finished and race_time_ms is not None and race_time_ms < self.time_ms:
            change = ("time", self.time_ms)
            self.entry.update(time_ms=race_time_ms, time_run=run_id, time_episode=episode)
        elif not finished and distance_m is not None and distance_m > self.distance_m:
            change = ("distance", self.distance_m)
        if distance_m is not None and distance_m > self.distance_m:
            self.entry.update(distance_m=round(float(distance_m), 3), distance_run=run_id)
        if change:
            self._save()
        return change

    def _read(self):
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self):
        # Re-read so other maps' entries (and other containers' writes) survive.
        data = self._read()
        data[self.map_name] = self.entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(self.path)
