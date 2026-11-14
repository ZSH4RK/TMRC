"""Write your own learning loop: a complete, small example (hill climbing).

This is NOT reinforcement learning in the SAC sense. It's one of the simplest
ways to improve a policy by trial and error, and it shows every piece you need
to plug in any method you like (evolution, CMA-ES, PPO, imitation, ...):

    1. create a Run         -> everything gets recorded under results/
    2. start a game         -> launch_games()
    3. wrap the env         -> RecordingEnv(TMNFEnv(...)) records every episode
    4. loop: reset / step   -> your algorithm decides what to try next

The policy is linear: action = tanh(W @ [obs, 1]) (see linear_agent.py).
Each iteration adds random noise to the best weights so far, drives one
episode, and keeps the new weights if they scored higher. The physics and
the policy are both deterministic, so one episode is a fair test of a set of
weights.

Run it (TM_SCRIPT picks the file, TM_ITERATIONS how long):
    TM_MODE=script TM_SCRIPT=agent/examples/hill_climb.py TM_ITERATIONS=200 docker compose up --build

Then drive the result with examples/linear_agent.py.
"""

import os
import sys
from pathlib import Path

# Make the agent/ folder importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import env as tmnf_env  # noqa: E402
import track_features  # noqa: E402
from env import TMNFEnv  # noqa: E402
from examples.linear_agent import act  # noqa: E402  (the same policy maths linear_agent.py drives with)
from experiments import RecordingEnv, Run  # noqa: E402
from game.instance_launch import hide_console, launch_games  # noqa: E402

ITERATIONS = int(os.environ.get("TM_ITERATIONS", "100"))
NOISE = float(os.environ.get("TM_NOISE", "0.1"))   # how big each random change is
BASE_PORT = int(os.environ.get("TM_BASE_PORT", "8775"))
MAP_FILE = f"{track_features.MAP_NAME}.Challenge.Gbx"


def drive_episode(env, weights):
    """Drive one episode with these weights. Returns (total reward, info)."""
    obs = env.reset()
    total, done, info = 0.0, False, {}
    while not done:
        obs, reward, done, info = env.step(act(weights, obs))
        total += reward
    return total, info


def main():
    env_config = tmnf_env.describe(MAP_FILE)
    run = Run.create(
        "train",
        {"trainer": "examples/hill_climb", "policy": "linear-hill-climb",
         "iterations": ITERATIONS, "noise": NOISE, "env": env_config},
        map_name=track_features.MAP_NAME, map_file=MAP_FILE, tracks_dir=track_features.TRACKS_DIR,
    )
    rng = np.random.default_rng(run.manifest["seed"])  # seeded: the run can be repeated

    (port,) = launch_games(1, "/home/wineuser/.wine-instances-example", BASE_PORT, "tmloader-example")
    env = RecordingEnv(TMNFEnv(port=port, map_file=MAP_FILE), run, policy="linear-hill-climb", env_config=env_config)

    # Start from "drive straight, full throttle": steer row all 0, throttle bias 1.
    best = np.zeros((2, track_features.STATE_DIM + 1))
    best[1, -1] = 1.0
    status = "completed"
    try:
        best_score, info = drive_episode(env, best)
        hide_console(0)
        for iteration in range(ITERATIONS):
            candidate = best + NOISE * rng.standard_normal(best.shape)
            score, info = drive_episode(env, candidate)
            improved = score > best_score
            if improved:
                best, best_score = candidate, score
                np.save(run.path / "best_weights.npy", best)
            run.log_metrics("hill_climb", {"iteration": iteration, "score": round(score, 2),
                                           "best_score": round(best_score, 2), "improved": improved,
                                           "done_reason": info.get("done_reason")})
            print(f"[hill_climb] {iteration}: score {score:.1f} (best {best_score:.1f}) {info.get('done_reason')}", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        env.close()
        np.save(run.path / "best_weights.npy", best)
        run.finalize(status, best_score=best_score if "best_score" in locals() else None)


if __name__ == "__main__":
    main()
