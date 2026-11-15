"""The long-range ahead input, the finish-approach reward, and loading a
checkpoint trained on a different observation."""

import pytest
import torch

import env as tmnf_env  # noqa: E402
import track_features as tf  # noqa: E402
from sac.networks import ActorNetwork  # noqa: E402

GAME_STATE = {"position": (944.0, 9.4, 400.0), "velocity": (0.0, 0.0, -50.0), "direction": (0.0, 0.0, -1.0)}


def rays(ahead):
    return [(250.0, None) if angle == 0 else (5.0, None) for angle in tf.RAY_ANGLES] if ahead else \
        [(None, None)] * len(tf.RAY_ANGLES)


def test_ahead_input():
    state = tf.build_state(GAME_STATE, rays(True))
    assert len(state) == tf.STATE_DIM
    assert state[tf.RAY_ANGLES.index(0)] == 1.0  # the 100m ray saturates
    assert state[-1] == pytest.approx(250.0 / tf.MAX_AHEAD_DIST_M)
    assert tf.build_state(GAME_STATE, rays(False))[-1] == 1.0


def test_track_position_offset():
    end_x, _y, end_z = tf._mid_points[-1]
    progress, offset = tf.track_position(end_x + 16.0, end_z)
    assert progress == pytest.approx(tf.TRACK_LENGTH, abs=0.5)
    assert offset == pytest.approx(16.0, abs=0.5)
    assert tf.progress_along_track(end_x + 16.0, end_z) == progress


def test_shaped_progress(monkeypatch):
    monkeypatch.setattr(tmnf_env, "FINISH_APPROACH_M", 50.0)
    length = tf.TRACK_LENGTH
    assert tmnf_env.shaped_progress(100.0, 10.0) == 100.0  # before the approach
    assert tmnf_env.shaped_progress(length - 25.0, 10.0) == pytest.approx(length - 25.0 - 5.0)
    # At the finish, beside the gate: moving towards it gains reward.
    assert tmnf_env.shaped_progress(length, 4.0) > tmnf_env.shaped_progress(length, 16.0)
    monkeypatch.setattr(tmnf_env, "FINISH_APPROACH_M", 0.0)
    assert tmnf_env.shaped_progress(length, 16.0) == length


def test_finish_approach_in_reward_config():
    assert "finish_approach_m" in tmnf_env.reward_config()


def test_checkpoint_with_other_observation_is_refused(tmp_path):
    old = ActorNetwork(0.0, [tf.STATE_DIM - 1], [1.0, 1.0], name="actor", chkpt_dir=str(tmp_path), device="cpu")
    old.save_checkpoint()
    new = ActorNetwork(0.0, [tf.STATE_DIM], [1.0, 1.0], name="actor", chkpt_dir=str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="different observation"):
        new.load_checkpoint()
    same = ActorNetwork(0.0, [tf.STATE_DIM - 1], [1.0, 1.0], name="actor", chkpt_dir=str(tmp_path), device="cpu")
    same.load_checkpoint()
    assert torch.equal(same.fc1.weight, old.fc1.weight)
