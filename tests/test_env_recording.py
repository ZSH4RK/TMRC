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


def test_wall_penalty_and_hits(results):
    """Steps ending against a wall lose WALL_PENALTY; the episode row counts
    separate contacts and the share of steps on a wall."""
    walls = {3, 4, 7}   # two separate contacts, three steps

    def run_steps(wall_steps):
        game = FakeGame()
        env = make_env(game)
        wait = game.wait

        def wait_with_walls():
            state = wait()
            env._wall_contact = game.i in wall_steps   # what _wait_for_step reads from the game
            return state
        env._wait_for_step = wait_with_walls
        run = Run.create("eval", root=results)
        rec = RecordingEnv(env, run, policy="test", env_config=tmnf_env.describe())
        rec.recorder.start(env.last_game_state["position"])
        rec._episode_started = 0
        steps = [rec.step_inputs({"left": False, "right": False, "accelerate": True, "brake": False}) for _ in range(10)]
        row = rec.end_episode({"done_reason": "timeout", "race_time_ms": 10 * STEP, "distance_m": 1.0, "track_pct": 1.0})
        return [s[1] for s in steps], [s[3]["wall_contact"] for s in steps], row

    clean, _, _ = run_steps(set())
    walled, contact, row = run_steps(walls)
    assert contact == [i + 1 in walls for i in range(10)]
    for i, (a, b) in enumerate(zip(clean, walled)):
        assert b == pytest.approx(a - (tmnf_env.WALL_PENALTY if i + 1 in walls else 0))
    assert row["wall_hits"] == 2 and row["wall_contact_pct"] == 30.0


def test_best_score_tied_to_reward(tmp_path):
    from experiments import best_run
    path = tmp_path / "best_avg_score.json"
    old = dict(best_run.LEGACY_REWARD)
    new = {**old, "wall_penalty": 1.0}
    path.write_text('{"avg_score": 1482.4}')        # saved before scores recorded their reward
    assert best_run.reward_matches(path, old) and best_run.load_score(path, "avg_score", old) == 1482.4
    assert not best_run.reward_matches(path, new) and best_run.load_score(path, "avg_score", new) == float("-inf")
    best_run.save_score(path, "avg_score", 900.0, reward=new)
    assert best_run.load_score(path, "avg_score", new) == 900.0
    assert best_run.load_score(path, "avg_score", old) == float("-inf")
