"""Persists "best score so far" thresholds and the single best-scoring
completed episode's trajectory, across process restarts.

Both save_models() (model checkpoint, gated on best rolling-average score)
and the best-run recording below (gated on best single-episode score) reset
their in-memory best-score baseline to -inf every time train_sac.py /
train_sac_vec.py starts. Without persisting that baseline too, a resumed
training session would treat its very first episode as beating -inf and
silently overwrite a genuinely better checkpoint/recording from a previous
session — the model/run files surviving a container rebuild (see
docker-compose.yml's bind mount) is only useful if the score that earned them
survives too.

The recorded trajectory itself is not a native TrackMania .Replay.Gbx file —
that's a proprietary binary format this project has no solid ground-truth
for producing. What's saved is enough to inspect the run
(positions/velocities/rewards over time) or play it back programmatically
(feed the same actions to a fresh TMNFEnv) — not to open in-game.
"""

import json
from pathlib import Path

import numpy as np


def load_score(score_path: Path, key: str) -> float:
    """The score a new result must beat before overwriting whatever score_path
    is guarding. -inf if nothing's been saved yet."""
    if not score_path.exists():
        return -np.inf
    return json.loads(score_path.read_text())[key]


def save_score(score_path: Path, key: str, value: float, extra: dict = None):
    score_path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: float(value)}
    if extra:
        data.update(extra)
    score_path.write_text(json.dumps(data))


def save_best_run(npz_path: Path, score_path: Path, states, actions, rewards, score: float, info: dict):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
    )
    save_score(score_path, "score", score, extra={"done_reason": info.get("done_reason"), "steps": len(rewards)})
