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


# The reward every score was saved under before scores recorded their reward
# settings (env.reward_config()): no wall penalty.
LEGACY_REWARD = {"reward_per_m_advanced": 1.0, "reward_per_ms": -6 / 5000, "finish_bonus": 100.0, "wall_penalty": 0.0}


def reward_matches(score_path: Path, reward: dict) -> bool:
    """Whether the score saved at score_path was earned under these reward
    settings. A score under a different reward isn't comparable: a harsher
    reward lowers every score, so the old one might never be beaten."""
    if not score_path.exists():
        return True
    return json.loads(score_path.read_text()).get("reward", LEGACY_REWARD) == reward


def load_score(score_path: Path, key: str, reward: dict = None) -> float:
    """The score a new result must beat before overwriting whatever score_path
    is guarding. -inf if nothing's been saved yet, or (given the current
    reward settings) if it was saved under different ones."""
    if not score_path.exists():
        return -np.inf
    if reward is not None and not reward_matches(score_path, reward):
        return -np.inf
    return json.loads(score_path.read_text())[key]


def save_score(score_path: Path, key: str, value: float, extra: dict = None, reward: dict = None):
    score_path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: float(value)}
    if extra:
        data.update(extra)
    if reward is not None:
        data["reward"] = reward
    score_path.write_text(json.dumps(data))


def save_best_run(npz_path: Path, score_path: Path, states, actions, rewards, score: float, info: dict,
                  reward: dict = None):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
    )
    save_score(score_path, "score", score, extra={"done_reason": info.get("done_reason"), "steps": len(rewards)},
               reward=reward)
