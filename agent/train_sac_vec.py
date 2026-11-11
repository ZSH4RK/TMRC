"""Multi-instance SAC training: N TMNF game instances feeding one shared
SACAgent. One training process, N game instances — see ../Plan.md's overall
architecture.

Each game runs on its own thread, so no game waits on another game or on a
reset. Learning runs in its own process (see sac/background.py), so it doesn't
hold up the games or compete with them for the GIL.

Each instance gets its own cloned Wine prefix (see instance_launch.py)
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
TM_EVAL_ONLY=true drives the loaded policy without learning or saving models.
"""

import json
import os
import signal
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import best_run
import track_features
from env import MAX_EPISODE_MS, TMNFEnv
from instance_launch import clone_prefix, hide_console, instance_tmnf_dir, patch_port, render_env, start_display
from sac.background import LearnerProcess, SharedPolicy, get_context
from sac.buffers import SharedReplayBuffer
from tm_launch import start_tmnf

MASTER_PREFIX = Path(os.environ["WINEPREFIX"])
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
REPLAY_SIZE = 100_000
TMP_DIR = Path(__file__).parent / "tmp"
CHKPT_DIR = str(TMP_DIR / "sac")
SNAPSHOT_DIR = str(TMP_DIR / "sac_latest")
# Same files train_sac.py (single-instance) uses — either trainer can resume
# past the other's best score/run, since both are bind-mounted in
# docker-compose.yml and survive a rebuild.
BEST_AVG_SCORE_FILE = TMP_DIR / "sac" / "best_avg_score.json"
PB_FILE = TMP_DIR / "pb.json"
BEST_RUN_NPZ = TMP_DIR / "best_run.npz"
BEST_RUN_SCORE_FILE = TMP_DIR / "best_run_score.json"

BOOT_STAGGER_S = 5
SNAPSHOT_INTERVAL_S = 600
PERF_INTERVAL_S = 30
# A rolling average of fewer episodes than this is mostly luck, and saving on
# it would set a best score that later, better averages struggle to beat.
AVG_WINDOW = 100
THREAD_JOIN_TIMEOUT_S = 10


def launch_instance(index: int) -> int:
    """Clone this instance's Wine prefix, patch its port, launch TMLoader in it. Returns the port."""
    port = BASE_PORT + index
    instance_dir = INSTANCES_DIR / str(index)
    clone_prefix(MASTER_PREFIX, instance_dir)
    patch_port(instance_dir, port)

    start_tmnf(
        tmnf_dir=instance_tmnf_dir(instance_dir),
        wineprefix=instance_dir,
        log_name=f"tmloader-train-{index}.log",
        display=start_display(index),
        extra_env=render_env(index),
    )
    return port


class Trainer:
    """State shared by the per-game threads. Episode bookkeeping runs under self.lock."""

    def __init__(self, policy: SharedPolicy, memory: SharedReplayBuffer, learner):
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
        # from a previous session (see best_run.py).
        self.best_score = best_run.load_score(BEST_AVG_SCORE_FILE, "avg_score")
        self.best_run_score = best_run.load_score(BEST_RUN_SCORE_FILE, "score")
        print(f"Resuming best avg_score={self.best_score}, best single-run score={self.best_run_score}")

        # Personal bests: fastest finish (ms) and furthest distance (m) along the
        # track, persisted so a restart doesn't re-announce old records.
        pb = json.loads(PB_FILE.read_text()) if PB_FILE.exists() else {}
        self.best_time_ms = pb.get("time_ms") or float("inf")
        self.best_distance_m = pb.get("distance_m", 0.0)
        print(f"[PB] Resuming PB time={self.best_time_ms}ms, furthest distance={self.best_distance_m:.1f}m")

    def run_instance(self, index: int, env: TMNFEnv):
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
            best_run.save_best_run(BEST_RUN_NPZ, BEST_RUN_SCORE_FILE, states, actions, rewards, score, info)

        pb_changed = False
        dist = info.get("distance_m", 0.0)
        if info.get("done_reason") == "finished":
            t = info["race_time_ms"]
            if t < self.best_time_ms:
                prev = "none" if self.best_time_ms == float("inf") else f"{self.best_time_ms / 1000:.2f}s"
                print(f"[PB] NEW PB: {t / 1000:.2f}s (prev {prev}) instance {index}, episode {episode}", flush=True)
                self.best_time_ms, pb_changed = t, True
        elif dist > self.best_distance_m:
            print(f"[PB] NEW FURTHEST: {dist:.0f}m = {info['track_pct']:.1f}% of track (prev {self.best_distance_m:.0f}m) instance {index}, episode {episode}", flush=True)
            pb_changed = True
        self.best_distance_m = max(self.best_distance_m, dist)
        if pb_changed:
            PB_FILE.parent.mkdir(parents=True, exist_ok=True)
            PB_FILE.write_text(json.dumps({"time_ms": None if self.best_time_ms == float("inf") else self.best_time_ms, "distance_m": self.best_distance_m}))

        print(
            f"[instance {index}] episode {episode} score {score:.1f} "
            f"avg_score {avg_score:.1f} dist={dist:.0f}m time={info['race_time_ms'] / 1000:.2f}s "
            f"done_reason={info.get('done_reason')}",
            flush=True,
        )


def raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    # `docker stop` sends SIGTERM; handle it like Ctrl-C so shutdown saves a snapshot.
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    # One thread per action is plenty, and leaves the cores for the games.
    torch.set_num_threads(1)

    print(f"=== TMRC SAC Vector Trainer Starting ({NUM_INSTANCES} instances) ===")

    ports = []
    for i in range(NUM_INSTANCES):
        print(f"[train_sac_vec] Launching instance {i} (port {BASE_PORT + i})...")
        ports.append(launch_instance(i))
        if i < NUM_INSTANCES - 1:
            time.sleep(BOOT_STAGGER_S)

    # Scales with N: launching instances is staggered, and this many Wine/GL
    # processes starting up concurrently is slower than the single-instance
    # 20s baseline (see agent.py/train_sac.py) at higher N.
    wait_s = 20 + 5 * NUM_INSTANCES
    print(f"Waiting {wait_s}s for {NUM_INSTANCES} games and TMInterface instances to start...")
    time.sleep(wait_s)

    envs = [TMNFEnv(port=p, map_file=MAP_FILE, game_speed=GAME_SPEED) for p in ports]
    print(f"Game speed: {GAME_SPEED}x, decision every {track_features.STEP_MS}ms, episode cutoff: {MAX_EPISODE_MS / 1000:.0f}s of race time")

    input_dims = [track_features.STATE_DIM]
    max_action = np.array([1.0, 1.0], dtype=np.float32)
    load_dir = {"latest": SNAPSHOT_DIR, "true": CHKPT_DIR}.get(LOAD_CHECKPOINT)

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
            "load_dir": load_dir,
            "updates_per_step": UPDATES_PER_STEP,
            "torch_threads": TORCH_THREADS,
        })
    trainer = Trainer(policy, memory, learner)

    threads = [
        threading.Thread(target=trainer.run_instance, args=(i, env), name=f"game-{i}", daemon=True)
        for i, env in enumerate(envs)
    ]
    for thread in threads:
        thread.start()

    started = last_perf = last_snapshot = time.monotonic()
    last_steps = 0
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
                last_perf, last_steps = now, trainer.steps
            if learner is not None and now - last_snapshot >= SNAPSHOT_INTERVAL_S:
                learner.save_snapshot()
                last_snapshot = now
    except KeyboardInterrupt:
        print("\nShutdown requested")
    finally:
        trainer.stop.set()
        for thread in threads:
            thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
        if learner is not None:
            learner.stop()
        print("Vector trainer shutting down gracefully")

    for error in trainer.errors:
        print(f"[train_sac_vec] {error}")
    if trainer.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
