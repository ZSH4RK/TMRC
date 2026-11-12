import numpy as np

from experiments.replay import EpisodeRecorder, Replay, compare_trajectories

NONE = {"left": False, "right": False, "accelerate": False, "brake": False}


def record(steps):
    """steps: list of input dicts, one per 50ms step starting at race time 0."""
    rec = EpisodeRecorder(meta={"env": {"step_ms": 50}})
    rec.start((0.0, 0.0, 0.0))
    for i, inputs in enumerate(steps):
        rec.add(input_time_ms=50 * i, race_time_ms=50 * (i + 1), inputs={**NONE, **inputs},
                position=(float(i), 0.0, 0.0), velocity=(10.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                reward=1.0, action=[0.0, 1.0], obs=[0.5] * 14, agent_ms=2.0 + i)
    return rec


def test_round_trip(tmp_path):
    rec = record([{"accelerate": True}, {"accelerate": True, "left": True}, {}])
    replay = rec.replay({"done_reason": "timeout"})
    path = replay.save(tmp_path / "ep1")
    assert path.name == "ep1.npz"
    assert (tmp_path / "ep1.inputs.txt").exists()

    loaded = Replay.load(path)
    assert len(loaded) == 3
    assert loaded.meta["done_reason"] == "timeout"
    assert loaded.meta["env"] == {"step_ms": 50}
    assert loaded.inputs_at(1) == {"left": True, "right": False, "accelerate": True, "brake": False}
    np.testing.assert_array_equal(loaded.position, replay.position)
    assert loaded.obs.shape == (3, 14) and loaded.action.shape == (3, 2)


def test_tmi_script_merges_held_keys():
    rec = record([{"accelerate": True}, {"accelerate": True, "left": True}, {"accelerate": True}, {"brake": True}])
    script = rec.replay().to_tmi_script()
    assert script.splitlines() == [
        "0-150 press up",
        "50-100 press left",
        "150-200 press down",
    ]
    assert rec.replay().to_tmi_script(offset_ms=10).splitlines()[0] == "10-160 press up"


def test_episode_stats():
    stats = record([{}, {}, {}]).episode_stats()
    assert stats["steps"] == 3
    assert stats["score"] == 3.0
    assert stats["mean_speed_kmh"] == 36.0
    assert stats["agent_ms_max"] == 4.0
    assert stats["step_gaps"] == 0


def test_compare_trajectories():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    assert compare_trajectories(a, a.copy(), 0.01)["match"]

    b = a.copy()
    b[2, 0] += 0.5
    result = compare_trajectories(a, b, 0.01)
    assert not result["match"]
    assert result["first_divergence_step"] == 2
    assert abs(result["max_deviation_m"] - 0.5) < 1e-9

    short = compare_trajectories(a, a[:2], 0.01)
    assert not short["match"] and short["compared_steps"] == 2
