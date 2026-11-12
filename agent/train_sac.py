"""SAC training loop for TMNF.

Adapted from Soft-Actor-Critic/main_sac.py: the same
choose_action -> step -> remember -> learn loop, driven here by TMNFEnv
(env.py) instead of a gym/pybullet env, with TMLoader/TmForever started the
same way agent.py starts it for a single debugging run.

Run inside the container (or on the host against a running instance) with:
    uv run agent/train_sac.py

Trained checkpoints land in agent/tmp/sac/ — my_agent.py loads the actor
checkpoint from there for inference. agent/tmp/ is bind-mounted in
docker-compose.yml, so both that and the best-run recording below survive a
container rebuild.

Every session is also recorded as a run under results/ (see experiments/).
"""

import os
import time
from pathlib import Path

import numpy as np

import best_run
import env as tmnf_env
import track_features
from env import MAX_EPISODE_MS, TMNFEnv
from experiments import RecordingEnv, Run
from sac.agent import SACAgent
from tm_launch import start_tmnf

PORT = int(os.environ.get("TMI_PORT", 8775))
# Defaults to track_features.MAP_NAME (TM_MAP_NAME) + ".Challenge.Gbx" so the
# loaded map and the wall/centerline CSV loaded by track_features stay in
# sync; override TM_MAP_FILE directly if the .Gbx filename differs.
MAP_FILE = os.environ.get("TM_MAP_FILE", f"{track_features.MAP_NAME}.Challenge.Gbx")
N_GAMES = int(os.environ.get("TM_N_GAMES", 2000))
LOAD_CHECKPOINT = os.environ.get("TM_LOAD_CHECKPOINT", "false").lower() == "true"
# Drive the loaded policy without learning or saving models.
EVAL_ONLY = os.environ.get("TM_EVAL_ONLY", "false").lower() == "true"
# A rolling average of fewer episodes than this is mostly luck, and saving on
# it would set a best score that later, better averages struggle to beat.
AVG_WINDOW = 100
TMP_DIR = Path(__file__).parent / "tmp"
CHKPT_DIR = str(TMP_DIR / "sac")
BEST_AVG_SCORE_FILE = TMP_DIR / "sac" / "best_avg_score.json"
BEST_RUN_NPZ = TMP_DIR / "best_run.npz"
BEST_RUN_SCORE_FILE = TMP_DIR / "best_run_score.json"


def main():
    print("=== TMRC SAC Trainer Starting ===")
    print(f"TMNF_DIR: {os.environ.get('TMNF_DIR', 'NOT SET')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")

    env_config = tmnf_env.describe(MAP_FILE)
    run = Run.create(
        "train",
        {"trainer": "train_sac", "policy": "sac-eval-only" if EVAL_ONLY else "sac-train", "n_games": N_GAMES,
         "load_checkpoint": LOAD_CHECKPOINT, "eval_only": EVAL_ONLY, "avg_window": AVG_WINDOW, "env": env_config},
        map_name=track_features.MAP_NAME, map_file=MAP_FILE, tracks_dir=track_features.TRACKS_DIR,
        loaded_checkpoint=CHKPT_DIR if LOAD_CHECKPOINT else None,
    )

    start_tmnf(log_name="tmloader-train.log")

    # We do NOT use a raw TCP port check here — the plugin's Render() detects
    # every TCP connect as a real client, calls OnConnect(), and waits 2000ms
    # for a handshake that never comes from a port-check close.
    print("Waiting 20s for game and TMInterface to start...")
    time.sleep(20)

    env = RecordingEnv(TMNFEnv(port=PORT, map_file=MAP_FILE), run,
                       policy=run.manifest["config"]["policy"], env_config=env_config)
    print(f"Game speed: {env.game_speed}x, episode cutoff: {MAX_EPISODE_MS / 1000:.0f}s of race time")
    agent = SACAgent(
        input_dims=[track_features.STATE_DIM],
        n_actions=2,
        max_action=np.array([1.0, 1.0], dtype=np.float32),
        chkpt_dir=CHKPT_DIR,
    )

    if LOAD_CHECKPOINT:
        agent.load_models()

    # Loaded from disk rather than starting at -inf every run: otherwise a
    # resumed session's very first episode would "beat" an in-memory -inf
    # baseline and silently overwrite a genuinely better checkpoint/recording
    # from a previous session (see best_run.py).
    best_score = best_run.load_score(BEST_AVG_SCORE_FILE, "avg_score")
    best_run_score = best_run.load_score(BEST_RUN_SCORE_FILE, "score")
    print(f"Resuming best avg_score={best_score}, best single-run score={best_run_score}")
    score_history = []
    status = "completed"

    try:
        for i in range(N_GAMES):
            obs = env.reset()
            done = False
            score = 0.0
            info = {}
            states, actions, rewards = [obs], [], []
            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                score += reward
                agent.remember(obs, action, reward, obs_, done)
                if not EVAL_ONLY:
                    agent.learn()
                states.append(obs_)
                actions.append(action)
                rewards.append(reward)
                obs = obs_

            score_history.append(score)
            avg_score = float(np.mean(score_history[-AVG_WINDOW:]))

            if not EVAL_ONLY and len(score_history) >= AVG_WINDOW and avg_score > best_score:
                best_score = avg_score
                agent.save_models()
                best_run.save_score(BEST_AVG_SCORE_FILE, "avg_score", best_score)

            if score > best_run_score:
                best_run_score = score
                best_run.save_best_run(BEST_RUN_NPZ, BEST_RUN_SCORE_FILE, states[:-1], actions, rewards, score, info)

            print(f"episode {i} score {score:.1f} avg_score {avg_score:.1f} done_reason={info.get('done_reason')}")
    except KeyboardInterrupt:
        print("\nShutdown requested")
        status = "interrupted"
    except Exception:
        status = "crashed"
        raise
    finally:
        env.close()
        if not EVAL_ONLY:
            run.archive_checkpoint(CHKPT_DIR, "best")
        run.finalize(status)
        print("Trainer shutting down gracefully")


if __name__ == "__main__":
    main()
