"""Evaluate any policy under fixed, recorded conditions.

Unlike training scores (exploration noise, a policy that changes every
update), an evaluation drives one fixed policy deterministically for a set
number of episodes and records the result as a run under results/, so it can
be compared with any other evaluation run on the same map and settings.

The physics are deterministic, so repeated episodes of a deterministic policy
should come out identical. summary.json's `distinct_outcomes` should be 1;
more means something in the setup isn't deterministic.

Run inside the container with (or TM_MODE=eval through docker compose):
    TM_EVAL_AGENT=sac:tmp/sac TM_EVAL_EPISODES=5 uv run agent/evaluate.py

TM_EVAL_AGENT takes any spec experiments.policies.load_policy() accepts:
sac:<checkpoint dir>, agent:<file.py> (a decide() entry, like my_agent.py),
forward, random.
"""

import os
import signal
import threading
import traceback

import env as tmnf_env
import track_features
from env import TMNFEnv
from experiments import RecordingEnv, Run, load_policy
from experiments.policies import policy_provenance
from game.instance_launch import hide_console, launch_games

EVAL_AGENT = os.environ.get("TM_EVAL_AGENT", "sac:tmp/sac")
EPISODES = int(os.environ.get("TM_EVAL_EPISODES", "5"))
NUM_INSTANCES = max(1, min(int(os.environ.get("TM_NUM_INSTANCES", "1")), EPISODES))
BASE_PORT = int(os.environ.get("TM_BASE_PORT", "8775"))
GAME_SPEED = float(os.environ.get("TM_GAME_SPEED", str(tmnf_env.DEFAULT_GAME_SPEED)))
MAP_FILE = os.environ.get("TM_MAP_FILE", f"{track_features.MAP_NAME}.Challenge.Gbx")
INSTANCES_DIR = "/home/wineuser/.wine-instances-eval"
THREAD_JOIN_TIMEOUT_S = 10


def raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


class Evaluation:
    def __init__(self, run: Run, first_policy, env_config: dict):
        self.run = run
        self.first_policy = first_policy
        self.env_config = env_config
        self.lock = threading.Lock()
        self.claimed = 0
        self.stop = threading.Event()
        self.errors = []

    def claim_episode(self) -> bool:
        with self.lock:
            if self.claimed >= EPISODES or self.stop.is_set():
                return False
            self.claimed += 1
            return True

    def run_instance(self, index: int, port: int):
        env = None
        try:
            # Instance 0 reuses the policy loaded up front; the others get
            # their own, so a stateful decide() can't leak between games.
            policy = self.first_policy if index == 0 else load_policy(EVAL_AGENT, seed=self.run.manifest["seed"] + index)
            env = RecordingEnv(TMNFEnv(port=port, map_file=MAP_FILE, game_speed=GAME_SPEED),
                               self.run, instance=index, policy=policy.label, env_config=self.env_config)
            console_hidden = False
            while self.claim_episode():
                obs = env.reset()
                if not console_hidden:
                    hide_console(index)
                    console_hidden = True
                done = False
                while not done and not self.stop.is_set():
                    inputs = policy.act(env.last_game_state, env.last_wall_distances, obs)
                    obs, _reward, done, info = env.step_inputs(inputs)
                if done:
                    row = info["episode_row"]
                    print(f"[eval {index}] episode {row['episode']}: {row['done_reason']} "
                          f"time={row['race_time_ms'] / 1000:.2f}s dist={row['distance_m']:.0f}m "
                          f"({row['track_pct']:.1f}%)", flush=True)
        except Exception:
            if not self.stop.is_set():
                self.errors.append(f"instance {index}: {traceback.format_exc()}")
                self.stop.set()
        finally:
            if env is not None:
                env.close()


def main():
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    print(f"=== TMRC Evaluation: {EVAL_AGENT}, {EPISODES} episode(s) on {MAP_FILE}, {NUM_INSTANCES} instance(s) ===")

    # Load the policy before starting any game, so a bad spec fails in seconds.
    policy = load_policy(EVAL_AGENT)
    env_config = tmnf_env.describe(MAP_FILE, GAME_SPEED)
    run = Run.create(
        "eval",
        {"policy": policy.label, "policy_spec": EVAL_AGENT, "policy_files": policy_provenance(policy)["files"],
         "episodes": EPISODES, "instances": NUM_INSTANCES, "env": env_config},
        map_name=track_features.MAP_NAME, map_file=MAP_FILE, tracks_dir=track_features.TRACKS_DIR,
        loaded_checkpoint=getattr(policy, "checkpoint_dir", None),
    )
    if hasattr(policy, "rng"):
        policy.rng.seed(run.manifest["seed"])

    evaluation = Evaluation(run, policy, env_config)
    status = "completed"
    threads = []
    try:
        ports = launch_games(NUM_INSTANCES, INSTANCES_DIR, BASE_PORT, "tmloader-eval")
        threads = [threading.Thread(target=evaluation.run_instance, args=(i, p), name=f"eval-{i}", daemon=True)
                   for i, p in enumerate(ports)]
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():
                thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\nShutdown requested")
        status = "interrupted"
    finally:
        evaluation.stop.set()
        for thread in threads:
            thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
        for error in evaluation.errors:
            print(f"[evaluate] {error}")
        if evaluation.errors:
            status = "crashed"
        run.finalize(status, errors=evaluation.errors or None)

    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
