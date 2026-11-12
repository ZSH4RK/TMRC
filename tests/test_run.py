import json
import threading

import numpy as np
import pytest

import compare_runs
from experiments import Run
from experiments.replay import EpisodeRecorder
from experiments.stats import read_episodes

NONE = {"left": False, "right": False, "accelerate": True, "brake": False}


def a_replay(steps=3):
    rec = EpisodeRecorder()
    rec.start((0, 0, 0))
    for i in range(steps):
        rec.add(input_time_ms=50 * i, race_time_ms=50 * (i + 1), inputs=NONE,
                position=(i, 0, 0), velocity=(1, 0, 0), direction=(0, 0, 1))
    return rec.replay()


def episode(done, time_ms, dist):
    return {"done_reason": done, "race_time_ms": time_ms, "distance_m": dist, "track_pct": dist / 10, "instance": 0}


def test_create_writes_manifest_and_code(results, monkeypatch):
    monkeypatch.setenv("TM_SEED", "123")
    monkeypatch.setenv("TM_STEP_MS", "50")
    run = Run.create("eval", {"policy": "forward"}, name="my test", root=results)
    m = json.loads((run.path / "manifest.json").read_text())
    assert run.path.name.endswith("-eval-my-test")
    assert m["seed"] == 123 and m["status"] == "running"
    assert m["env"]["TM_STEP_MS"] == "50"
    assert m["code"]["files"]["agent/env.py"]
    assert (run.path / "code" / "agent" / "experiments" / "run.py").exists()
    import os
    assert os.environ["TM_RUN_DIR"] == str(run.path)


def test_seed_is_applied(results, monkeypatch):
    monkeypatch.setenv("TM_SEED", "7")
    Run.create("eval", root=results)
    first = np.random.rand()
    Run.create("eval", root=results)
    assert np.random.rand() == first


def test_best_replay_policy(results):
    run = Run.create("train", root=results, save_replays="best")
    rows = [
        run.log_episode(episode("stuck", 3000, 100), a_replay()),     # best so far
        run.log_episode(episode("stuck", 3000, 50), a_replay()),      # worse
        run.log_episode(episode("finished", 40000, 1000), a_replay()),  # best
        run.log_episode(episode("finished", 45000, 1000), a_replay()),  # slower
    ]
    assert [bool(r.get("replay")) for r in rows] == [True, False, True, False]
    assert rows[2]["replay"].endswith("-finished-40000ms.npz")
    assert (run.path / rows[2]["replay"]).exists()

    summary = run.finalize()
    assert summary["episodes"] == 4 and summary["best_time_ms"] == 40000
    assert json.loads((run.path / "manifest.json").read_text())["status"] == "completed"


def test_finished_and_none_policies(results):
    run = Run.create("train", root=results, save_replays="finished")
    run.log_episode(episode("finished", 40000, 1000), a_replay())
    assert run.log_episode(episode("finished", 45000, 1000), a_replay()).get("replay")
    off = Run.create("train", root=results, save_replays="none")
    assert not off.log_episode(episode("finished", 1, 1), a_replay()).get("replay")
    with pytest.raises(ValueError):
        Run.create("train", root=results, save_replays="sometimes")


def test_threads_share_a_run(results):
    run = Run.create("train", root=results, save_replays="none")

    def worker(i):
        for _ in range(50):
            run.log_episode({**episode("stuck", 3000, 10), "instance": i})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    run.finalize()
    rows = read_episodes(run.path)
    assert len(rows) == 200
    assert sorted(r["episode"] for r in rows) == list(range(1, 201))


def test_attached_writers_get_their_own_file(results, monkeypatch):
    owner = Run.create("drive", root=results)
    monkeypatch.setenv("TM_INSTANCE_INDEX", "1")
    child = Run.attach()
    child.log_episode(episode("finished", 30000, 1000))
    child.finalize()
    owner.log_episode(episode("stuck", 3000, 10))
    summary = owner.finalize("interrupted")
    assert (owner.path / "episodes-instance-1.csv").exists()
    assert summary["episodes"] == 2 and summary["status"] == "interrupted"


def test_context_manager_marks_crash(results):
    with pytest.raises(RuntimeError):
        with Run.create("train", root=results) as run:
            raise RuntimeError("boom")
    summary = json.loads((run.path / "summary.json").read_text())
    assert summary["status"] == "crashed" and "boom" in summary["error"]


def test_metrics_and_checkpoints(results, tmp_path):
    run = Run.create("train", root=results)
    run.log_metrics("perf", {"steps": 10, "steps_per_s": 5.0})
    run.log_metrics("perf", {"steps": 20, "steps_per_s": 6.0})
    lines = (run.path / "metrics" / "perf.csv").read_text().splitlines()
    assert lines[0] == "wall_time,elapsed_s,steps,steps_per_s" and len(lines) == 3

    chkpt = tmp_path / "sac"
    chkpt.mkdir()
    (chkpt / "actor_sac").write_bytes(b"weights")
    run.archive_checkpoint(chkpt, "best")
    assert (run.path / "checkpoints" / "best" / "actor_sac").read_bytes() == b"weights"
    assert "actor_sac" in json.loads((run.path / "manifest.json").read_text())["checkpoints"]["best"]


def test_compare_runs_cli(results, capsys, monkeypatch):
    monkeypatch.setenv("TM_SEED", "1")
    a = Run.create("eval", {"policy": "forward", "env": {"step_ms": 50}}, root=results)
    a.log_episode(episode("finished", 40000, 1000))
    a.finalize()
    monkeypatch.setenv("TM_SEED", "2")
    b = Run.create("eval", {"policy": "forward", "env": {"step_ms": 20}}, root=results, name="b")
    b.log_episode(episode("finished", 35000, 1000))  # left running: no summary.json yet
    capsys.readouterr()

    compare_runs.main(["--root", str(results)])
    out = capsys.readouterr().out.splitlines()
    assert b.id in out[1] and a.id in out[2]  # faster run ranks first

    compare_runs.main(["--root", str(results), "--diff", a.id, "-b"])
    out = capsys.readouterr().out
    assert "seed: 1 -> 2" in out
    assert "env.step_ms: 50 -> 20" in out
    assert "[code files] identical" in out
