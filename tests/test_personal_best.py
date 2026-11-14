"""results/pb.json: per-map personal bests."""

import json

from experiments.personal_best import PersonalBest


def test_time_and_distance(tmp_path):
    pb = PersonalBest(tmp_path, "A03-Race", "sha1")
    assert pb.update("run1", 1, False, 9000, 300.0) == ("distance", 0.0)
    assert pb.update("run1", 2, False, 9000, 200.0) is None
    assert pb.update("run1", 3, True, 40000, 1466.0) == ("time", float("inf"))
    assert pb.update("run1", 4, True, 41000, 1466.0) is None
    data = json.loads((tmp_path / "pb.json").read_text())["A03-Race"]
    assert data["time_ms"] == 40000 and data["time_run"] == "run1" and data["time_episode"] == 3
    assert data["distance_m"] == 1466.0


def test_maps_kept_apart_and_reloaded(tmp_path):
    PersonalBest(tmp_path, "A03-Race", "sha1").update("r", 1, True, 30000, 1466.0)
    PersonalBest(tmp_path, "A01-Race", "sha2").update("r", 1, False, 5000, 100.0)
    a03 = PersonalBest(tmp_path, "A03-Race", "sha1")
    assert a03.time_ms == 30000 and a03.distance_m == 1466.0
    assert PersonalBest(tmp_path, "A01-Race", "sha2").time_ms == float("inf")


def test_track_change_drops_distance_keeps_time(tmp_path):
    PersonalBest(tmp_path, "A03-Race", "old").update("r", 1, True, 25440, 1493.8)
    pb = PersonalBest(tmp_path, "A03-Race", "new")
    assert pb.time_ms == 25440 and pb.distance_m == 0.0
    assert pb.update("r2", 1, False, 9000, 800.0) == ("distance", 0.0)
