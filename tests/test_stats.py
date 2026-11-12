from experiments.stats import is_better, rank_key, summarize


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
