"""Per-step episode recording, replay files, and trajectory comparison.

A Replay stores the inputs sent to the game at each decision step plus the
car's trajectory that resulted. The inputs are what reproduce a run: TMNF's
physics are deterministic, so sending the same inputs at the same race times
should give the same trajectory, and verify_replay.py checks exactly that.

This is not a native TrackMania .Replay.Gbx (see best_run.py for why). Each
saved replay also gets a TMInterface input script (*.inputs.txt) for loading
the run into the game by hand. TMInterface accepts `<from>-<to> press <key>`
lines with times in ms. Check one in-game before relying on the exact timing:
the inputs are recorded at the race time they were sent, and whether the game
applies them on that tick or the next one hasn't been measured here.
`offset_ms` shifts the whole script if it turns out to be the next one.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

INPUT_KEYS = ("left", "right", "accelerate", "brake")
# TMInterface's names for the same keys.
TMI_KEYS = {"left": "left", "right": "right", "accelerate": "up", "brake": "down"}
REPLAY_FORMAT_VERSION = 1


@dataclass
class Replay:
    input_time_ms: np.ndarray   # (N,) race time at which each step's inputs were sent
    race_time_ms: np.ndarray    # (N,) race time after the step
    inputs: np.ndarray          # (N, 4) uint8, columns INPUT_KEYS
    position: np.ndarray        # (N, 3) after the step
    velocity: np.ndarray        # (N, 3)
    direction: np.ndarray       # (N, 3)
    reward: np.ndarray          # (N,)
    start_position: np.ndarray  # (3,) at the start of the run
    action: Optional[np.ndarray] = None  # (N, A) continuous policy output, when there is one
    obs: Optional[np.ndarray] = None     # (N, D) observation the step's decision was made from
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.race_time_ms)

    def save(self, path: Path) -> Path:
        """Write <path>.npz plus <path>.inputs.txt. Returns the .npz path."""
        path = Path(path).with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "input_time_ms": self.input_time_ms, "race_time_ms": self.race_time_ms,
            "inputs": self.inputs, "position": self.position, "velocity": self.velocity,
            "direction": self.direction, "reward": self.reward, "start_position": self.start_position,
        }
        if self.action is not None:
            arrays["action"] = self.action
        if self.obs is not None:
            arrays["obs"] = self.obs
        meta = {"format_version": REPLAY_FORMAT_VERSION, **self.meta}
        npz_path = path.parent / (path.name + ".npz")
        np.savez_compressed(npz_path, meta=np.array(json.dumps(meta, default=str)), **arrays)
        (path.parent / (path.name + ".inputs.txt")).write_text(self.to_tmi_script())
        return npz_path

    @classmethod
    def load(cls, path: Path) -> "Replay":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            optional = {k: data[k] for k in ("action", "obs") if k in data.files}
            return cls(
                input_time_ms=data["input_time_ms"], race_time_ms=data["race_time_ms"],
                inputs=data["inputs"], position=data["position"], velocity=data["velocity"],
                direction=data["direction"], reward=data["reward"], start_position=data["start_position"],
                meta=meta, **optional,
            )

    def inputs_at(self, i: int) -> dict:
        return {key: bool(self.inputs[i, k]) for k, key in enumerate(INPUT_KEYS)}

    def to_tmi_script(self, offset_ms: int = 0) -> str:
        """TMInterface input script: one `<from>-<to> press <key>` line per
        stretch of time a key was held."""
        n = len(self)
        if n == 0:
            return ""
        # A step's inputs are held until the next step's are sent; the last
        # step's until the end of the run.
        ends = np.append(self.input_time_ms[1:], self.race_time_ms[-1])
        lines = []
        for k, key in enumerate(INPUT_KEYS):
            held = self.inputs[:, k].astype(bool)
            i = 0
            while i < n:
                if not held[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and held[j + 1]:
                    j += 1
                start = max(0, int(self.input_time_ms[i]) + offset_ms)
                end = int(ends[j]) + offset_ms
                if end > start:
                    lines.append((start, f"{start}-{end} press {TMI_KEYS[key]}"))
                i = j + 1
        return "\n".join(line for _, line in sorted(lines)) + "\n"


class EpisodeRecorder:
    """Accumulates one episode step by step. Call start() after each reset,
    add() after each step, finish() when it ends."""

    def __init__(self, meta: Optional[dict] = None, keep_obs: bool = True):
        self.meta = dict(meta or {})
        self.keep_obs = keep_obs
        self.start(None)

    def start(self, start_position, start_time_ms: int = 0):
        self.start_position = tuple(start_position) if start_position is not None else (np.nan,) * 3
        self.start_time_ms = start_time_ms
        self._input_time, self._race_time, self._inputs = [], [], []
        self._pos, self._vel, self._dir, self._reward = [], [], [], []
        self._action, self._obs = [], []
        self._agent_ms = []
        self.step_gaps = 0

    def add(self, *, input_time_ms, race_time_ms, inputs, position, velocity, direction,
            reward=0.0, action=None, obs=None, agent_ms=None, step_gap=False):
        self._input_time.append(input_time_ms)
        self._race_time.append(race_time_ms)
        self._inputs.append([bool(inputs[key]) for key in INPUT_KEYS])
        self._pos.append(position)
        self._vel.append(velocity)
        self._dir.append(direction)
        self._reward.append(reward)
        if action is not None:
            self._action.append(np.asarray(action, dtype=np.float32))
        if obs is not None and self.keep_obs:
            self._obs.append(np.asarray(obs, dtype=np.float32))
        if agent_ms is not None:
            self._agent_ms.append(agent_ms)
        self.step_gaps += int(bool(step_gap))

    def __len__(self):
        return len(self._race_time)

    def replay(self, extra_meta: Optional[dict] = None) -> Replay:
        n = len(self)
        return Replay(
            input_time_ms=np.array(self._input_time, dtype=np.int32),
            race_time_ms=np.array(self._race_time, dtype=np.int32),
            inputs=np.array(self._inputs, dtype=np.uint8).reshape(n, len(INPUT_KEYS)),
            position=np.array(self._pos, dtype=np.float32).reshape(n, 3),
            velocity=np.array(self._vel, dtype=np.float32).reshape(n, 3),
            direction=np.array(self._dir, dtype=np.float32).reshape(n, 3),
            reward=np.array(self._reward, dtype=np.float32),
            start_position=np.array(self.start_position, dtype=np.float32),
            action=np.stack(self._action) if len(self._action) == n and n else None,
            obs=np.stack(self._obs) if len(self._obs) == n and n else None,
            meta={**self.meta, **(extra_meta or {})},
        )

    def episode_stats(self) -> dict:
        """Driving and health stats for the episode's CSV row."""
        speeds = np.linalg.norm(np.array(self._vel, dtype=np.float64).reshape(-1, 3), axis=1) * 3.6 if self._vel else np.array([])
        return {
            "steps": len(self),
            "score": float(np.sum(self._reward)) if self._reward else 0.0,
            "mean_speed_kmh": round(float(speeds.mean()), 2) if speeds.size else None,
            "max_speed_kmh": round(float(speeds.max()), 2) if speeds.size else None,
            "step_gaps": self.step_gaps,
            "agent_ms_mean": round(float(np.mean(self._agent_ms)), 3) if self._agent_ms else None,
            "agent_ms_max": round(float(np.max(self._agent_ms)), 3) if self._agent_ms else None,
        }


def compare_trajectories(expected: np.ndarray, actual: np.ndarray, tolerance_m: float) -> dict:
    """How far `actual` positions strayed from `expected`, step by step."""
    expected = np.asarray(expected, dtype=np.float64).reshape(-1, 3)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1, 3)
    n = min(len(expected), len(actual))
    deviation = np.linalg.norm(expected[:n] - actual[:n], axis=1) if n else np.array([])
    over = np.nonzero(deviation > tolerance_m)[0]
    return {
        "expected_steps": int(len(expected)),
        "actual_steps": int(len(actual)),
        "compared_steps": int(n),
        "max_deviation_m": float(deviation.max()) if n else None,
        "mean_deviation_m": float(deviation.mean()) if n else None,
        "first_divergence_step": int(over[0]) if over.size else None,
        "tolerance_m": tolerance_m,
        "match": bool(n == len(expected) == len(actual) and not over.size),
    }
