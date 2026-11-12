"""Tests for the recording/replay framework. They run on the host without
the game: TMInterface is replaced by fakes where needed.

    uv run pytest
"""

import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    # Run.create exports TM_RUN_DIR for child processes; don't let it leak
    # between tests, or from the shell running them.
    for var in ("TM_RUN_DIR", "TM_SEED", "TM_RUN_NAME", "TM_SAVE_REPLAYS", "TM_INSTANCE_INDEX"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results"
