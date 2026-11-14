"""Re-drive a recorded replay and check the game reproduces it.

Sends the replay's recorded inputs to a fresh game, step by step, and compares
the car's position after every step with the recording. TMNF's physics are
deterministic, so a correct setup reproduces a run to within float noise. A
divergence points at something that would make every other result
unrepeatable too: a protocol desync, a missed step, different game files.

Run inside the container with (or TM_MODE=verify through docker compose):
    TM_REPLAY=results/<run>/replays/<file>.npz uv run agent/verify_replay.py

Writes <replay>.verify.json next to the replay and records the reproduced
episode as a run of kind "verify". Exits 0 on a match, 2 on a divergence.
"""

import json
import os
import signal
from pathlib import Path

import env as tmnf_env
import track_features
from env import TMNFEnv
from experiments import RecordingEnv, Replay, Run
from experiments.provenance import PROJECT_DIR, sha256_file
from experiments.replay import compare_trajectories
from game.instance_launch import hide_console, launch_games

TOLERANCE_M = float(os.environ.get("TM_VERIFY_TOL_M", "0.01"))
BASE_PORT = int(os.environ.get("TM_BASE_PORT", "8775"))
INSTANCES_DIR = "/home/wineuser/.wine-instances-verify"
# Settings that must match the recording for a re-drive to mean anything.
MUST_MATCH = ("map_name", "map_file", "step_ms")


def raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def resolve_replay(value: str) -> Path:
    path = Path(value)
    for candidate in (path, PROJECT_DIR / path, PROJECT_DIR / "results" / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Replay not found: {value}")


def main():
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    replay_path = resolve_replay(os.environ["TM_REPLAY"])
    replay = Replay.load(replay_path)
    recorded_env = replay.meta.get("env", {})
    game_speed = float(recorded_env.get("game_speed", tmnf_env.DEFAULT_GAME_SPEED))
    map_file = recorded_env.get("map_file", f"{track_features.MAP_NAME}.Challenge.Gbx")
    env_config = tmnf_env.describe(map_file, game_speed)

    mismatched = {k: (recorded_env.get(k), env_config[k]) for k in MUST_MATCH
                  if k in recorded_env and recorded_env[k] != env_config[k]}
    if mismatched:
        raise SystemExit(f"Replay was recorded with different settings (recorded, now): {mismatched}. "
                         "Set TM_MAP_NAME / TM_STEP_MS to match.")

    print(f"=== TMRC Replay Verification: {replay_path} ({len(replay)} steps, tolerance {TOLERANCE_M}m) ===")
    run = Run.create(
        "verify",
        {"policy": "replay", "replay": str(replay_path), "replay_sha256": sha256_file(replay_path),
         "replay_meta": replay.meta, "tolerance_m": TOLERANCE_M, "env": env_config},
        map_name=track_features.MAP_NAME, map_file=map_file, tracks_dir=track_features.TRACKS_DIR,
        save_replays="all",
    )

    status, result = "completed", None
    env = None
    try:
        (port,) = launch_games(1, INSTANCES_DIR, BASE_PORT, "tmloader-verify")
        env = RecordingEnv(TMNFEnv(port=port, map_file=map_file, game_speed=game_speed),
                           run, instance=0, policy="replay", env_config=env_config)
        env.reset()
        hide_console(0)
        positions, timing_mismatches = [], 0
        done, info = False, {}
        for i in range(len(replay)):
            _obs, _reward, done, info = env.step_inputs(replay.inputs_at(i))
            positions.append(info["position"])
            timing_mismatches += int(info["input_time_ms"] != int(replay.input_time_ms[i]))
            if done:
                break
        if not done:
            env.end_episode({**info, "done_reason": "replay_end"})

        result = compare_trajectories(replay.position, positions, TOLERANCE_M)
        result.update({
            "replay": str(replay_path),
            "verify_run": run.id,
            "input_timing_mismatches": timing_mismatches,
            "recorded": {"done_reason": replay.meta.get("done_reason"), "race_time_ms": replay.meta.get("race_time_ms")},
            "reproduced": {"done_reason": info.get("done_reason"), "race_time_ms": info.get("race_time_ms")},
        })
        result["match"] = bool(result["match"] and timing_mismatches == 0
                               and result["recorded"]["race_time_ms"] in (None, result["reproduced"]["race_time_ms"]))
        replay_path.with_suffix(".verify.json").write_text(json.dumps(result, indent=2))
        verdict = "MATCH" if result["match"] else "DIVERGED"
        print(f"[verify] {verdict}: max deviation {result['max_deviation_m']}m over {result['compared_steps']} steps, "
              f"first divergence at step {result['first_divergence_step']}, "
              f"recorded {result['recorded']} vs reproduced {result['reproduced']}", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as e:
        status = "crashed"
        raise SystemExit(f"[verify] {type(e).__name__}: {e}")
    finally:
        if env is not None:
            env.close()
        run.finalize(status, verification=result)

    if result is None or not result["match"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
