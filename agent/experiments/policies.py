"""One driving interface for everything that can be evaluated.

A policy is any object with
    act(game_state, wall_distances, obs) -> {left, right, accelerate, brake}
and a `label` naming it in results. load_policy() builds one from a spec:

    sac:<checkpoint dir>   a trained SAC actor, driven deterministically
                           (mean action, no exploration noise)
    agent:<file.py>        any Python file with a decide(game_state,
                           wall_distances) function: the interface my_agent.py
                           and competition entries implement
    forward                baseline: accelerate, never steer
    random                 baseline: random inputs (seeded)

Every policy is built fresh per game instance, so a stateful decide() never
shares state between instances.
"""

import importlib.util
import itertools
import random
import sys
from pathlib import Path

from . import provenance

AGENT_DIR = provenance.AGENT_DIR
_module_counter = itertools.count()


class SACPolicy:
    def __init__(self, checkpoint_dir: str):
        import torch

        import track_features
        from sac.networks import ActorNetwork

        self._torch = torch
        self._to_inputs = track_features.action_to_inputs
        self.checkpoint_dir = str(_resolve(checkpoint_dir))
        self.actor = ActorNetwork(0.0, [track_features.STATE_DIM], [1.0, 1.0], n_actions=2,
                                  chkpt_dir=self.checkpoint_dir, device="cpu")
        self.actor.load_checkpoint()
        self.actor.eval()
        self.label = f"sac:{self.checkpoint_dir}"
        self.files = [Path(self.actor.checkpoint_file)]

    def act(self, game_state, wall_distances, obs):
        state = self._torch.tensor(obs, dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            action = self.actor.deterministic_action(state)[0].numpy()
        return self._to_inputs(action)


class AgentFilePolicy:
    def __init__(self, path: str):
        self.path = _resolve(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Agent file not found: {self.path}")
        # A unique module name per load, so each instance gets its own module
        # (and so its own globals) even when loading the same file twice.
        name = f"tmrc_agent_{self.path.stem}_{next(_module_counter)}"
        spec = importlib.util.spec_from_file_location(name, self.path)
        module = importlib.util.module_from_spec(spec)
        # Let the entry import modules sitting next to it.
        sys.path.insert(0, str(self.path.parent))
        spec.loader.exec_module(module)
        if not hasattr(module, "decide"):
            raise AttributeError(f"{self.path} has no decide(game_state, wall_distances) function")
        self.module = module
        self.label = f"agent:{self.path.name}"
        self.files = [self.path]

    def act(self, game_state, wall_distances, obs):
        inputs = self.module.decide(game_state, wall_distances)
        return {key: bool(inputs.get(key, False)) for key in ("left", "right", "accelerate", "brake")}


class ForwardPolicy:
    label = "forward"
    files = []

    def act(self, game_state, wall_distances, obs):
        return {"left": False, "right": False, "accelerate": True, "brake": False}


class RandomPolicy:
    label = "random"
    files = []

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, game_state, wall_distances, obs):
        steer = self.rng.choice((-1, 0, 1))
        return {"left": steer < 0, "right": steer > 0, "accelerate": self.rng.random() < 0.8, "brake": self.rng.random() < 0.1}


def _resolve(path) -> Path:
    """Paths are relative to the agent directory unless absolute, matching how
    the trainers name checkpoint dirs (tmp/sac)."""
    path = Path(path)
    return path if path.is_absolute() else (AGENT_DIR / path)


def load_policy(spec: str, seed=None):
    kind, _, arg = spec.partition(":")
    if kind == "sac":
        return SACPolicy(arg or "tmp/sac")
    if kind == "agent":
        return AgentFilePolicy(arg or "my_agent.py")
    if kind == "forward":
        return ForwardPolicy()
    if kind == "random":
        return RandomPolicy(seed)
    raise ValueError(f"Unknown policy spec {spec!r}: use sac:<dir>, agent:<file.py>, forward or random")


def policy_provenance(policy) -> dict:
    """Hashes of the files a policy was loaded from, for the manifest."""
    return {
        "label": policy.label,
        "files": {str(p): provenance.sha256_file(p) for p in getattr(policy, "files", [])},
    }
