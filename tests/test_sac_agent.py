"""SACAgent's optional entropy tuning and learning-rate decay, on random data."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sac.agent import SACAgent  # noqa: E402


def make_agent(tmp_path, **kwargs):
    agent = SACAgent(input_dims=[4], n_actions=2, max_size=1000, batch_size=32,
                     chkpt_dir=str(tmp_path / "sac"), **kwargs)
    rng = np.random.default_rng(0)
    for _ in range(100):
        agent.remember(rng.normal(size=4), rng.uniform(-1, 1, size=2), rng.normal(), rng.normal(size=4), False)
    return agent


def test_defaults_match_original(tmp_path):
    agent = make_agent(tmp_path)
    for _ in range(5):
        stats = agent.learn()
    assert float(stats["temperature"]) == 1.0
    assert float(stats["lr"]) == pytest.approx(0.0003)


def test_auto_entropy_moves_temperature(tmp_path):
    # A target far above the policy's entropy: the temperature has to rise.
    agent = make_agent(tmp_path, auto_entropy=True, target_entropy=10.0)
    for _ in range(20):
        stats = agent.learn()
    assert float(stats["temperature"]) > 1.0


def test_lr_decay(tmp_path):
    agent = make_agent(tmp_path, lr_decay_updates=10, lr_final=0.1)
    for _ in range(20):
        stats = agent.learn()
    assert float(stats["lr"]) == pytest.approx(0.00003)
    assert agent.critic_1.optimizer.param_groups[0]["lr"] == pytest.approx(0.00003)


def test_temperature_and_updates_survive_save(tmp_path):
    agent = make_agent(tmp_path, auto_entropy=True, target_entropy=10.0)
    for _ in range(20):
        agent.learn()
    agent.save_snapshot(str(tmp_path / "snap"))
    resumed = make_agent(tmp_path, auto_entropy=True)
    resumed.load_models(str(tmp_path / "snap"))
    assert resumed.updates == 20
    assert resumed.log_temperature.item() == pytest.approx(agent.log_temperature.item())


def test_old_checkpoint_without_state_loads(tmp_path):
    agent = make_agent(tmp_path)
    agent.save_snapshot(str(tmp_path / "snap"))
    (tmp_path / "snap" / "sac_state").unlink()
    resumed = make_agent(tmp_path, auto_entropy=True)
    resumed.load_models(str(tmp_path / "snap"))
    assert resumed.updates == 0 and resumed.log_temperature.item() == 0.0
