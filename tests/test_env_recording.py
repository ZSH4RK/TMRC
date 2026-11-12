"""TMNFEnv.step_inputs -> RecordingEnv, with TMInterface replaced by a fake.
Checks the info contract the recorder depends on, and that a recorded episode
reproduces the inputs that were sent."""

from unittest import mock

import pytest

track_features = pytest.importorskip("track_features", reason="needs the Tracks/ CSVs")
import env as tmnf_env  # noqa: E402
from experiments import RecordingEnv, Replay, Run  # noqa: E402
from experiments.stats import read_episodes  # noqa: E402

STEP = track_features.STEP_MS


class FakeGame:
    """Feeds _wait_for_step a car driving along the track's centreline."""

    def __init__(self, gap_at=None):
        self.race_time = 0
        self.i = 0
        self.gap_at = gap_at
        self.points = track_features._mid_points

    def wait(self):
        self.i += 1
        self.race_time += STEP * (2 if self.i == self.gap_at else 1)
        x, y, z = self.points[min(self.i, len(self.points) - 1)]
        state = {"position": (x, y, z), "velocity": (20.0, 0.0, 0.0), "direction": (1.0, 0.0, 0.0)}
        return state, self.race_time


def make_env(game):
    env = tmnf_env.TMNFEnv(port=0)
    env.tm = mock.Mock()
    env._wait_for_step = game.wait
    # What reset() leaves behind, without the socket dance.
    first = {"position": game.points[0], "velocity": (0.0, 0.0, 0.0), "direction": (1.0, 0.0, 0.0)}
    env._prev_progress = track_features.progress_along_track(first["position"][0], first["position"][2])
    env._prev_race_time = 0
    env.last_race_time = 0
    env._build_obs(first)
    return env


def test_step_info_contract():
    game = FakeGame(gap_at=3)
    env = make_env(game)
    infos = [env.step_inputs({"left": True, "right": False, "accelerate": True, "brake": False})[3] for _ in range(4)]
    env.tm.set_input_state.assert_called_with(left=True, right=False, accelerate=True, brake=False)
    assert [i["input_time_ms"] for i in infos] == [0, STEP, 2 * STEP, 4 * STEP]
    assert [i["step_gap"] for i in infos] == [False, False, True, False]
    assert infos[0]["inputs"]["accelerate"] is True
    assert infos[0]["position"] == game.points[1]
    assert env.last_wall_distances is not None and len(env.last_wall_distances) == len(track_features.RAY_ANGLES)


def test_recorded_episode(results):
    run = Run.create("eval", root=results, save_replays="all")
    env = RecordingEnv(make_env(FakeGame()), run, instance=2, policy="test", env_config=tmnf_env.describe())
    env.recorder.start(env.last_game_state["position"])
    env._episode_started = 0
    pattern = [{"left": i % 3 == 0, "right": False, "accelerate": True, "brake": i % 5 == 0} for i in range(20)]
    for inputs in pattern:
        env.step_inputs(inputs)
    row = env.end_episode({"done_reason": "timeout", "race_time_ms": 20 * STEP, "distance_m": 10.0, "track_pct": 1.0})
    run.finalize()

    assert row["instance"] == 2 and row["steps"] == 20
    replay = Replay.load(run.path / row["replay"])
    assert [replay.inputs_at(i) for i in range(20)] == pattern
    assert replay.meta["env"]["step_ms"] == STEP
    assert replay.meta["policy"] == "test"
    assert read_episodes(run.path)[0]["done_reason"] == "timeout"
