"""Plain driving agent: one game, driven by a decide() agent, recorded.

Drives TM_AGENT (default agent:my_agent.py; any spec
experiments.policies.load_policy() accepts, e.g. agent:<competitor>.py or
forward) for as long as it runs, restarting the race after each episode.

Episodes run through TMNFEnv, the same environment the trainers and
evaluate.py use, so they end the same way everywhere: at the finish, at
MAX_EPISODE_MS of race time, or after STUCK_MS without progress.

Every episode is recorded (see experiments/). Started by launcher.py, this
joins the launcher's run (TM_RUN_DIR) as instance TM_INSTANCE_INDEX; started
directly, it records a run of its own.
"""

import os
import signal
import sys
import time

import env as tmnf_env
import track_features
from env import TMNFEnv
from experiments import RecordingEnv, Run, load_policy
from experiments.policies import policy_provenance
from instance_launch import hide_console
from tm_launch import start_tmnf

# TMI_PORT is set per-instance by launcher.py when TM_NUM_INSTANCES > 1
# (instance i uses TM_BASE_PORT + i); must match this instance's `custom_port`.
PORT = int(os.environ.get("TMI_PORT", 8775))
GAME_SPEED = float(os.environ.get("TM_GAME_SPEED", "5.0"))
INSTANCE = int(os.environ.get("TM_INSTANCE_INDEX", "0"))
AGENT_SPEC = os.environ.get("TM_AGENT", "agent:my_agent.py")
# Must match track_features.MAP_NAME: that's the map whose wall/centerline
# CSVs the observations are built from.
MAP_FILE = os.environ.get("TM_MAP_FILE", f"{track_features.MAP_NAME}.Challenge.Gbx")


def raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    # launcher.py forwards SIGTERM; handle it like Ctrl-C so the run is closed properly.
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    print("=== TMRC Agent Starting ===")
    print(f"TMNF_DIR: {os.environ.get('TMNF_DIR', 'NOT SET')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")

    policy = load_policy(AGENT_SPEC)
    env_config = tmnf_env.describe(MAP_FILE, GAME_SPEED)
    run = Run.create_or_attach(
        "drive",
        {"policy": policy.label, "policy_spec": AGENT_SPEC, "policy_files": policy_provenance(policy)["files"],
         "instances": 1, "env": env_config},
        map_name=track_features.MAP_NAME, map_file=MAP_FILE, tracks_dir=track_features.TRACKS_DIR,
    )

    start_tmnf()
    # We do NOT use a raw TCP port check here — the plugin's Render() detects every
    # TCP connect as a real client, calls OnConnect(), and waits 2000ms for a
    # handshake that never comes from a port-check close.
    print("Waiting 20s for game and TMInterface to start...")
    time.sleep(20)

    print(f"Connecting to TMInterface on port {PORT}, driving {policy.label}")
    env = RecordingEnv(TMNFEnv(port=PORT, map_file=MAP_FILE, game_speed=GAME_SPEED),
                       run, instance=INSTANCE, policy=policy.label, env_config=env_config)
    status = "completed"
    try:
        obs = env.reset()
        hide_console(INSTANCE)
        print("✓ Connected and racing")
        while True:
            inputs = policy.act(env.last_game_state, env.last_wall_distances, obs)
            obs, _reward, done, info = env.step_inputs(inputs)
            if done:
                row = info["episode_row"]
                print(f"Episode {row['episode']}: {row['done_reason']} time={row['race_time_ms'] / 1000:.2f}s "
                      f"dist={row['distance_m']:.0f}m ({row['track_pct']:.1f}%)", flush=True)
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nShutdown requested")
        status = "interrupted"
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        status = "crashed"
    finally:
        env.close()
        run.finalize(status)
        print("Agent shutting down gracefully")

    if status == "crashed":
        sys.exit(1)


if __name__ == "__main__":
    main()
