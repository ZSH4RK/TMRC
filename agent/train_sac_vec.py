"""Multi-instance SAC training: N TMNF game instances feeding one shared
SACAgent. One training process, N game instances — see ../Plan.md's overall
architecture.

Each game runs on its own thread, so no game waits on another game or on a
reset. Learning runs in its own process (see sac/background.py), so it doesn't
hold up the games or compete with them for the GIL.

Each instance gets its own cloned Wine prefix (see game/instance_launch.py)
because TMInterface's port is read once at startup from a file inside the
prefix, and Wine scopes named objects to the wineserver tied to a given
WINEPREFIX — this is the same cloning launcher.py does for N parallel
*driving* instances; here it's N parallel instances feeding one *trainer*
instead of N separate agent.py processes.

Run inside the container with:
    TM_NUM_INSTANCES=5 uv run agent/train_sac_vec.py

Trained checkpoints land in agent/tmp/sac/ — the same place train_sac.py
(single-instance) and my_agent.py use, so either can pick up where the other
left off.

Stopping and resuming: agent/tmp/sac_latest/ always holds a recent snapshot
(see SNAPSHOT_INTERVAL_S), also written on shutdown. TM_LOAD_CHECKPOINT=true
resumes training from the best checkpoint, =latest from that snapshot. The
replay buffer isn't saved, so a resumed run starts by refilling it.
TM_EVAL_ONLY=true drives the loaded policy without learning or saving models
(for a proper, deterministic evaluation use evaluate.py instead).

Every session is recorded as a run under results/ (see experiments/): the
manifest, one row per episode, replays of improving episodes, perf and learner
metrics, and copies of the checkpoints it ended with.
"""

import os
import signal
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments import best_run
import env as tmnf_env
import track_features
from env import MAX_EPISODE_MS, TMNFEnv
from experiments import RecordingEnv, Run
from experiments.personal_best import PersonalBest
from game.instance_launch import hide_console, launch_games
from sac.background import LearnerProcess, SharedPolicy, get_context
from sac import settings as sac_settings
from sac.buffers import SharedReplayBuffer

INSTANCES_DIR = Path("/home/wineuser/.wine-instances-train")

NUM_INSTANCES = int(os.environ.get("TM_NUM_INSTANCES", "4"))
BASE_PORT = int(os.environ.get("TM_BASE_PORT", "8775"))
GAME_SPEED = float(os.environ.get("TM_GAME_SPEED", "5.0"))
MAP_FILE = os.environ.get("TM_MAP_FILE", f"{track_features.MAP_NAME}.Challenge.Gbx")
N_STEPS = int(os.environ.get("TM_N_STEPS", 2_000_000))
LOAD_CHECKPOINT = os.environ.get("TM_LOAD_CHECKPOINT", "false").lower()
EVAL_ONLY = os.environ.get("TM_EVAL_ONLY", "false").lower() == "true"
UPDATES_PER_STEP = float(os.environ.get("TM_UPDATES_PER_STEP", "1.0"))
# Small networks gain nothing from many threads, and the games need the cores.
TORCH_THREADS = int(os.environ.get("TM_TORCH_THREADS", "2"))
REPLAY_SIZE = sac_settings.REPLAY_SIZE
TMP_DIR = Path(__file__).parent / "tmp"
CHKPT_DIR = str(TMP_DIR / "sac")
SNAPSHOT_DIR = str(TMP_DIR / "sac_latest")
# Same files train_sac.py (single-instance) uses — either trainer can resume
# past the other's best score/run, since both are bind-mounted in
# docker-compose.yml and survive a rebuild.
BEST_AVG_SCORE_FILE = TMP_DIR / "sac" / "best_avg_score.json"
BEST_RUN_NPZ = TMP_DIR / "best_run.npz"
BEST_RUN_SCORE_FILE = TMP_DIR / "best_run_score.json"

BOOT_STAGGER_S = 5
SNAPSHOT_INTERVAL_S = 600
PERF_INTERVAL_S = 30
# A rolling average of fewer episodes than this is mostly luck, and saving on
# it would set a best score that later, better averages struggle to beat.
AVG_WINDOW = 100
THREAD_JOIN_TIMEOUT_S = 10


class Trainer:
    """State shared by the per-game threads. Episode bookkeeping runs under self.lock."""

    def __init__(self, policy: SharedPolicy, memory: SharedReplayBuffer, learner, run: Run):
        self.run = run
        self.policy = policy
        self.memory = memory
        self.learner = learner
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.errors = []
        self.steps = 0
        self.completed_scores = []

        # Loaded from disk rather than starting at -inf every run: otherwise a
        # resumed session's very first episode would "beat" an in-memory -inf
        # baseline and silently overwrite a genuinely better checkpoint/recording
        # from a previous session (see experiments/best_run.py).
        self.reward = tmnf_env.reward_config()
        self.best_score = best_run.load_score(BEST_AVG_SCORE_FILE, "avg_score", self.reward)
        self.best_run_score = best_run.load_score(BEST_RUN_SCORE_FILE, "score", self.reward)
        print(f"Resuming best avg_score={self.best_score}, best single-run score={self.best_run_score}")

        # Personal bests for this map: fastest finish and furthest distance,
        # kept in results/pb.json so a restart doesn't re-announce old records.
        self.pb = PersonalBest(run.path.parent, track_features.MAP_NAME,
                               run.manifest.get("game", {}).get("track_csv_sha256"))
        self.pb_lock = threading.Lock()
        print(f"[PB] {track_features.MAP_NAME}: best time={self.pb.time_ms}ms, "
              f"furthest distance={self.pb.distance_m:.1f}m ({self.pb.path})")

    def run_instance(self, index: int, env: RecordingEnv):
        """Drive one game until asked to stop. The body of each game thread."""
        try:
            obs = env.reset()
            hide_console(index)
            score = 0.0
            states, actions, rewards = [], [], []
            while not self.stop.is_set():
                action = self.policy.choose_action(obs)
                next_obs, reward, done, info = env.step(action)

                self.memory.store_transition(obs, action, reward, next_obs, done)
                score += reward
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                self.steps += 1  # only ever read for reporting, so a lost increment doesn't matter

                if done:
                    with self.lock:
                        self._end_episode(index, score, states, actions, rewards, info)
                    score = 0.0
                    states, actions, rewards = [], [], []
                    obs = env.reset()
                else:
                    obs = next_obs
        except Exception:
            if not self.stop.is_set():
                self.errors.append(f"instance {index}: {traceback.format_exc()}")
                self.stop.set()
        finally:
            env.close()

    def _end_episode(self, index, score, states, actions, rewards, info):
        self.completed_scores.append(score)
        episode = len(self.completed_scores)
        avg_score = float(np.mean(self.completed_scores[-AVG_WINDOW:]))

        if not EVAL_ONLY and episode >= AVG_WINDOW and avg_score > self.best_score:
            self.best_score = avg_score
            self.learner.save_best(self.best_score)

        if score > self.best_run_score:
            self.best_run_score = score
            best_run.save_best_run(BEST_RUN_NPZ, BEST_RUN_SCORE_FILE, states, actions, rewards, score, info,
                                   reward=self.reward)

        dist = info.get("distance_m", 0.0)
        finished = info.get("done_reason") == "finished"
        with self.pb_lock:
            change = self.pb.update(self.run.path.name, episode, finished, info.get("race_time_ms"), dist)
        if change and change[0] == "time":
            prev = "none" if change[1] == float("inf") else f"{change[1] / 1000:.2f}s"
            print(f"[PB] NEW PB: {info['race_time_ms'] / 1000:.2f}s (prev {prev}) instance {index}, episode {episode}", flush=True)
        elif change:
            print(f"[PB] NEW FURTHEST: {dist:.0f}m = {info['track_pct']:.1f}% of track (prev {change[1]:.0f}m) instance {index}, episode {episode}", flush=True)

        print(
            f"[instance {index}] episode {episode} score {score:.1f} "
            f"avg_score {avg_score:.1f} dist={dist:.0f}m time={info['race_time_ms'] / 1000:.2f}s "
            f"done_reason={info.get('done_reason')}",
            flush=True,
        )


def raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def keep_previous_best(run):
    """The saved best checkpoint's score was earned under a different reward
    (e.g. TM_WALL_PENALTY changed), so it can't be compared with this run's:
    the run starts its best score afresh and will overwrite agent/tmp/sac/ at
    its first improvement. Copy the old best into this run's folder first."""
    if not best_run.reward_matches(BEST_AVG_SCORE_FILE, tmnf_env.reward_config()):
        run.archive_checkpoint(CHKPT_DIR, "previous-best")
        print(f"[best] The saved best was scored under a different reward: copied it to "
              f"{run.path / 'checkpoints' / 'previous-best'}; best score starts afresh.", flush=True)


def main():
    # `docker stop` sends SIGTERM; handle it like Ctrl-C so shutdown saves a snapshot.
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    # One thread per action is plenty, and leaves the cores for the games.
    torch.set_num_threads(1)

    print(f"=== TMRC SAC Vector Trainer Starting ({NUM_INSTANCES} instances) ===")

    load_dir = {"latest": SNAPSHOT_DIR, "true": CHKPT_DIR}.get(LOAD_CHECKPOINT)
    env_config = tmnf_env.describe(MAP_FILE, GAME_SPEED)
    policy_label = "sac-eval-only" if EVAL_ONLY else "sac-train"
    run = Run.create(
        "train",
        {"trainer": "train_sac_vec", "policy": policy_label, "instances": NUM_INSTANCES, "n_steps": N_STEPS,
         "load_checkpoint": LOAD_CHECKPOINT, "eval_only": EVAL_ONLY, "updates_per_step": UPDATES_PER_STEP,
         "sac": sac_settings.describe(2), "avg_window": AVG_WINDOW, "torch_threads": TORCH_THREADS, "env": env_config},
        map_name=track_features.MAP_NAME, map_file=MAP_FILE, tracks_dir=track_features.TRACKS_DIR,
        loaded_checkpoint=load_dir,
    )

    keep_previous_best(run)

    # Scales with N: launching instances is staggered, and this many Wine/GL
    # processes starting up concurrently is slower than the single-instance
    # 20s baseline (see agent.py/train_sac.py) at higher N.
    ports = launch_games(NUM_INSTANCES, INSTANCES_DIR, BASE_PORT, "tmloader-train", stagger_s=BOOT_STAGGER_S)

    envs = [
        RecordingEnv(TMNFEnv(port=p, map_file=MAP_FILE, game_speed=GAME_SPEED), run,
                     instance=i, policy=policy_label, env_config=env_config)
        for i, p in enumerate(ports)
    ]
    print(f"Game speed: {GAME_SPEED}x, decision every {track_features.STEP_MS}ms, episode cutoff: {MAX_EPISODE_MS / 1000:.0f}s of race time")

    input_dims = [track_features.STATE_DIM]
    max_action = np.array([1.0, 1.0], dtype=np.float32)

    ctx = get_context()
    memory = SharedReplayBuffer(REPLAY_SIZE, input_dims, 2, ctx)
    policy = SharedPolicy(input_dims, 2, max_action, ctx, chkpt_dir=load_dir or CHKPT_DIR)
    if EVAL_ONLY:
        learner = None
        policy.actor.load_checkpoint()
    else:
        learner = LearnerProcess(ctx, memory, policy, {
            "input_dims": input_dims,
            "n_actions": 2,
            "max_action": max_action,
            "chkpt_dir": CHKPT_DIR,
            "snapshot_dir": SNAPSHOT_DIR,
            "best_score_file": BEST_AVG_SCORE_FILE,
            "reward": tmnf_env.reward_config(),
            "load_dir": load_dir,
            "updates_per_step": UPDATES_PER_STEP,
            "agent_kwargs": sac_settings.agent_kwargs(2),
            "torch_threads": TORCH_THREADS,
            # Distinct from the games' seed, so the learner's sampling isn't
            # a copy of the policy's.
            "seed": run.manifest["seed"] + 1000,
            "metrics_dir": str(run.path / "metrics"),
        })
    trainer = Trainer(policy, memory, learner, run)

    threads = [
        threading.Thread(target=trainer.run_instance, args=(i, env), name=f"game-{i}", daemon=True)
        for i, env in enumerate(envs)
    ]
    for thread in threads:
        thread.start()

    started = last_perf = last_snapshot = time.monotonic()
    last_steps = 0
    status = "completed"
    try:
        while not trainer.stop.is_set() and trainer.steps < N_STEPS:
            time.sleep(1)
            if learner is not None and not learner.is_alive():
                raise RuntimeError("Learner process died")

            now = time.monotonic()
            if now - last_perf >= PERF_INTERVAL_S:
                steps_per_s = (trainer.steps - last_steps) / (now - last_perf)
                game_speed = steps_per_s * track_features.STEP_MS / 1000 / NUM_INSTANCES
                updates = 0 if learner is None else learner.updates
                ratio = updates / max(trainer.steps, 1)
                print(
                    f"[perf] {steps_per_s:.0f} steps/s, {game_speed:.2f}x real time per game, "
                    f"{trainer.steps} steps, {updates} updates ({ratio:.2f} per step), "
                    f"{(now - started) / 60:.0f}min",
                    flush=True,
                )
                run.log_metrics("perf", {
                    "steps": trainer.steps, "steps_per_s": round(steps_per_s, 1),
                    "realtime_per_game": round(game_speed, 3), "updates": updates,
                    "updates_per_step": round(ratio, 3), "episodes": len(trainer.completed_scores),
                })
                last_perf, last_steps = now, trainer.steps
            if learner is not None and now - last_snapshot >= SNAPSHOT_INTERVAL_S:
                learner.save_snapshot()
                last_snapshot = now
    except KeyboardInterrupt:
        print("\nShutdown requested")
        status = "interrupted"
    except Exception:
        status = "crashed"
        raise
    finally:
        trainer.stop.set()
        for thread in threads:
            thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
        if learner is not None:
            learner.stop()
        # Keep the weights with the results: agent/tmp/ is overwritten by
        # the next session, the run directory isn't.
        if not EVAL_ONLY:
            run.archive_checkpoint(CHKPT_DIR, "best")
            run.archive_checkpoint(SNAPSHOT_DIR, "latest")
        for error in trainer.errors:
            print(f"[train_sac_vec] {error}")
        if trainer.errors:
            status = "crashed"
        run.finalize(status, steps=trainer.steps, updates=0 if learner is None else learner.updates,
                     errors=trainer.errors or None)
        print("Vector trainer shutting down gracefully")

    if trainer.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
