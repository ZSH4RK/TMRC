"""Drives with weights learned by examples/hill_climb.py.

Shows how an agent can use the same observation vector the trainers use
(track_features.build_state) instead of raw game state, and load something it
learned from a file.

Try it (after hill_climb.py has saved weights):
    TM_MODE=eval TM_EVAL_AGENT=agent:examples/linear_agent.py \\
    TM_WEIGHTS=../results/<run_id>/best_weights.npy docker compose up --build

TM_WEIGHTS is relative to the agent/ folder, or absolute.
"""

import os
from pathlib import Path

import numpy as np

import track_features

_weights = None  # loaded on first use, so hill_climb.py can import act() without a weights file


def load_weights():
    """Weights saved by hill_climb.py: shape (2, STATE_DIM + 1), steer row then throttle row."""
    path = Path(os.environ.get("TM_WEIGHTS", "examples/linear_weights.npy"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return np.load(path)


def act(weights, obs):
    """Linear policy: action = tanh(weights @ [obs, 1]). The trailing 1 lets
    each output have a constant offset (a bias). Shared with hill_climb.py,
    so training and driving compute actions the same way."""
    return np.tanh(weights @ np.append(obs, 1.0))


def decide(game_state, wall_distances):
    global _weights
    if _weights is None:
        _weights = load_weights()
    obs = np.array(track_features.build_state(game_state, wall_distances), dtype=np.float32)
    return track_features.action_to_inputs(act(_weights, obs))
