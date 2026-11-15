"""results/<run>/pb.json: a run's personal bests."""

import json

from experiments.personal_best import PersonalBest


def test_time_and_distance(tmp_path):
    pb = PersonalBest(tmp_path, "A03-Race")
    assert pb.update(1, False, 9000, 300.0) == ("distance", 0.0)
    assert pb.update(2, False, 9000, 200.0) is None
    assert pb.update(3, True, 40000, 1466.0) == ("time", float("inf"))
    assert pb.update(4, True, 41000, 1466.0) is None
    data = json.loads((tmp_path / "pb.json").read_text())
    assert data == {"map_name": "A03-Race", "time_ms": 40000, "time_episode": 3,
                    "distance_m": 1466.0, "distance_episode": 3}


def test_each_run_starts_afresh(tmp_path):
    PersonalBest(tmp_path / "run1", "A03-Race").update(1, True, 30000, 1466.0)
    pb = PersonalBest(tmp_path / "run2", "A03-Race")
    assert pb.time_ms == float("inf") and pb.distance_m == 0.0
    assert pb.update(1, True, 35000, 1466.0) == ("time", float("inf"))
    assert json.loads((tmp_path / "run1" / "pb.json").read_text())["time_ms"] == 30000
