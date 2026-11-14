from experiments.stats import input_stats, is_better, medal, rank_key, summarize


def row(done, time_ms, dist, **kw):
    return {"done_reason": done, "finished": done == "finished", "race_time_ms": time_ms,
            "distance_m": dist, "track_pct": dist / 10, "score": dist, "step_gaps": 0, "agent_ms_max": 1.0, **kw}


def test_rank_is_race_order():
    fast = row("finished", 30_000, 1000)
    slow = row("finished", 40_000, 1000)
    far = row("timeout", 120_000, 900)
    near = row("stuck", 5_000, 100)
    assert sorted([near, slow, far, fast], key=rank_key) == [fast, slow, far, near]
    assert is_better(fast, None)
    assert not is_better(far, slow)


def test_summarize():
    rows = [row("stuck", 5_000, 100), row("finished", 40_000, 1000), row("finished", 30_000, 1000)]
    s = summarize(rows, window=2)
    assert s["episodes"] == 3
    assert s["finishes"] == 2
    assert s["best_time_ms"] == 30_000
    assert s["median_time_ms"] == 35_000
    assert s["done_reasons"] == {"stuck": 1, "finished": 2}
    assert s["last_window"]["episodes"] == 2
    assert s["best_episode"]["race_time_ms"] == 30_000
    assert s["distinct_outcomes"] == 3


def test_summarize_empty():
    assert summarize([]) == {"episodes": 0}


A03_TIMES = {"bronze_ms": 28610, "silver_ms": 22290, "gold_ms": 19840, "author_ms": 18750}


def test_medal():
    assert medal(27980, A03_TIMES) == {"medal": "bronze", "gap_to_author_ms": 9230}
    assert medal(19840, A03_TIMES)["medal"] == "gold"      # equal to a medal time earns it
    assert medal(18000, A03_TIMES) == {"medal": "author", "gap_to_author_ms": -750}
    assert medal(30000, A03_TIMES)["medal"] == "none"
    assert medal(None, A03_TIMES) == {"medal": None, "gap_to_author_ms": None}
    assert medal(27980, None) == {"medal": None, "gap_to_author_ms": None}


def test_input_stats():
    #         left   right  accel  brake
    rows = [(False, False, True, False),
            (True, False, True, False),
            (True, False, False, True),
            (False, True, True, False)]
    stats = input_stats(rows, race_ms=200)
    assert stats == {"accelerate_pct": 75.0, "brake_pct": 25.0, "steer_pct": 75.0,
                     # straight -> left -> left -> right: 2 changes in 0.2 s
                     "steer_changes_per_s": 10.0}
    assert input_stats([], 0)["steer_changes_per_s"] is None
