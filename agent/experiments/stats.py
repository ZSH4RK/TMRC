"""Episode summaries and the ranking used to call one result better than another.

Results are ranked like a race, not by reward: a finish beats any non-finish,
faster finishes beat slower ones, and among non-finishes the one that got
further wins. Reward depends on the shaping constants in env.py, so scores
from runs with different shaping can't be compared; race results can.
"""

import csv
import math
from collections import Counter
from pathlib import Path

# Column order of episodes*.csv. Rows may carry more keys; they're dropped.
EPISODE_FIELDS = [
    "episode", "instance", "wall_time", "policy", "done_reason", "finished",
    "race_time_ms", "medal", "gap_to_author_ms", "distance_m", "track_pct",
    "checkpoint_times_ms", "final_x", "final_y", "final_z", "score", "steps",
    "mean_speed_kmh", "max_speed_kmh", "accelerate_pct", "brake_pct", "steer_pct",
    "steer_changes_per_s", "wall_hits", "wall_contact_pct", "step_gaps", "agent_ms_mean", "agent_ms_max",
    "wall_s", "replay",
]

_NUMERIC = {
    "episode": int, "steps": int, "step_gaps": int, "race_time_ms": int, "gap_to_author_ms": int,
    "distance_m": float, "track_pct": float, "score": float, "mean_speed_kmh": float,
    "max_speed_kmh": float, "agent_ms_mean": float, "agent_ms_max": float, "wall_s": float,
    "final_x": float, "final_y": float, "final_z": float, "accelerate_pct": float,
    "brake_pct": float, "steer_pct": float, "steer_changes_per_s": float,
    "wall_hits": int, "wall_contact_pct": float,
}

MEDALS = ("author", "gold", "silver", "bronze")


def medal(race_time_ms, map_times) -> dict:
    """The medal a finish earned and its gap to the author time, from the
    map's times (Tracks/<map>/map.json). A finish equal to a medal time earns
    it, as in the game. Without a finish or the map's times, both are None."""
    if race_time_ms is None or not map_times:
        return {"medal": None, "gap_to_author_ms": None}
    earned = next((m for m in MEDALS if race_time_ms <= map_times[f"{m}_ms"]), "none")
    return {"medal": earned, "gap_to_author_ms": int(race_time_ms - map_times["author_ms"])}


def input_stats(inputs, race_ms) -> dict:
    """How the car was driven: % of steps accelerating, braking and steering,
    and how often the steering changed (left / straight / right) per second
    of race time. A high change rate is a twitchy, wobbling driver.
    inputs: one (left, right, accelerate, brake) row per step, over race_ms
    of race time."""
    rows = [tuple(bool(v) for v in row) for row in inputs]
    if not rows:
        return {"accelerate_pct": None, "brake_pct": None, "steer_pct": None, "steer_changes_per_s": None}
    n = len(rows)
    steer = [(1 if right else 0) - (1 if left else 0) for left, right, _, _ in rows]
    changes = sum(1 for a, b in zip(steer, steer[1:]) if a != b)
    return {
        "accelerate_pct": round(100 * sum(r[2] for r in rows) / n, 1),
        "brake_pct": round(100 * sum(r[3] for r in rows) / n, 1),
        "steer_pct": round(100 * sum(1 for s in steer if s) / n, 1),
        "steer_changes_per_s": round(1000 * changes / race_ms, 2) if race_ms and race_ms > 0 else None,
    }


def rank_key(row: dict) -> tuple:
    """Sort key where smaller is better."""
    if _truthy(row.get("finished")):
        return (0, float(row["race_time_ms"]))
    return (1, -float(row.get("distance_m") or 0.0))


def is_better(row: dict, best) -> bool:
    return best is None or rank_key(row) < rank_key(best)


def _truthy(value) -> bool:
    return value is True or str(value).lower() in ("true", "1")


def parse_row(row: dict) -> dict:
    out = dict(row)
    for key, cast in _NUMERIC.items():
        value = out.get(key)
        if value in (None, ""):
            out[key] = None
            continue
        try:
            out[key] = cast(float(value)) if cast is int else cast(value)
        except ValueError:
            out[key] = None
    out["finished"] = _truthy(out.get("finished"))
    return out


def read_episodes(run_dir: Path) -> list:
    """All episode rows of a run, from every writer's episodes*.csv."""
    rows = []
    for path in sorted(Path(run_dir).glob("episodes*.csv")):
        with open(path, newline="") as f:
            rows.extend(parse_row(r) for r in csv.DictReader(f))
    rows.sort(key=lambda r: (r.get("wall_time") or "", r.get("episode") or 0))
    return rows


def _mean_std_ci(values: list) -> dict:
    """Mean, sample std and a 95% normal-approximation confidence interval."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95": 1.96 * std / math.sqrt(n) if n > 1 else None}


def _median(values: list):
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def summarize(rows: list, window: int = 100) -> dict:
    """Aggregate a run's episodes. `last_window` covers the most recent
    `window` episodes, which is what matters for a run that was learning."""
    if not rows:
        return {"episodes": 0}

    def block(subset):
        finishes = [r["race_time_ms"] for r in subset if r["finished"] and r["race_time_ms"] is not None]
        return {
            "episodes": len(subset),
            "finishes": len(finishes),
            "finish_rate": len(finishes) / len(subset),
            "best_time_ms": min(finishes) if finishes else None,
            "median_time_ms": _median(finishes),
            "distance_m": _mean_std_ci([r["distance_m"] for r in subset if r["distance_m"] is not None]),
            "track_pct": _mean_std_ci([r["track_pct"] for r in subset if r["track_pct"] is not None]),
            "score": _mean_std_ci([r["score"] for r in subset if r["score"] is not None]),
            "done_reasons": dict(Counter(r.get("done_reason") or "unknown" for r in subset)),
            # Distinct outcomes of a deterministic policy on one map should be
            # 1: the physics are deterministic. More means something isn't.
            "distinct_outcomes": len({(r.get("done_reason"), r["race_time_ms"], round(r["distance_m"] or 0, 2)) for r in subset}),
        }

    best = min(rows, key=rank_key)
    summary = block(rows)
    summary["last_window"] = block(rows[-window:])
    summary["window"] = window
    summary["best_episode"] = {k: best.get(k) for k in ("episode", "instance", "done_reason", "race_time_ms", "distance_m", "track_pct", "replay")}
    summary["step_gaps"] = sum(r["step_gaps"] or 0 for r in rows)
    lat = [r["agent_ms_max"] for r in rows if r["agent_ms_max"] is not None]
    summary["agent_ms_max"] = max(lat) if lat else None
    return summary
