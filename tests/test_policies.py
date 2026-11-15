
import pytest

from experiments import load_policy

ENTRY = """
calls = 0

def decide(game_state, wall_distances):
    global calls
    calls += 1
    return {"accelerate": True, "left": calls % 2 == 0}
"""


def test_agent_file_policies_are_independent(tmp_path):
    path = tmp_path / "entry.py"
    path.write_text(ENTRY)
    a = load_policy(f"agent:{path}")
    b = load_policy(f"agent:{path}")
    assert a.label == "agent:entry.py"
    assert a.act({}, [], None) == {"left": False, "right": False, "accelerate": True, "brake": False}
    assert a.act({}, [], None)["left"] is True
    # b has its own module, so its own call counter.
    assert b.act({}, [], None)["left"] is False


def test_bad_specs(tmp_path):
    with pytest.raises(ValueError):
        load_policy("nonsense")
    with pytest.raises(FileNotFoundError):
        load_policy(f"agent:{tmp_path / 'missing.py'}")
    no_decide = tmp_path / "x.py"
    no_decide.write_text("pass\n")
    with pytest.raises(AttributeError):
        load_policy(f"agent:{no_decide}")


def test_baselines():
    assert load_policy("forward").act({}, [], None)["accelerate"]
    r1, r2 = load_policy("random", seed=3), load_policy("random", seed=3)
    assert [r1.act({}, [], None) for _ in range(20)] == [r2.act({}, [], None) for _ in range(20)]


def test_sac_policy_loads_checkpoint(tmp_path):
    import numpy as np

    import track_features
    from sac.networks import ActorNetwork
    ActorNetwork(0.0, [track_features.STATE_DIM], [1.0, 1.0], name="actor",
                 chkpt_dir=str(tmp_path), device="cpu").save_checkpoint()
    policy = load_policy(f"sac:{tmp_path}")
    inputs = policy.act({}, [], np.zeros(track_features.STATE_DIM, dtype=np.float32))
    assert set(inputs) == {"left", "right", "accelerate", "brake"}
