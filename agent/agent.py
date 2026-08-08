import os
import sys

from tminterface2 import TMInterface, MessageType
import struct

import time
import subprocess
import threading
from pathlib import Path

import my_agent

sys.path.insert(0, str(Path(__file__).parent.parent / "Tracks"))
from interpolate_track import TrackInterpolator, Vec3

PORT = int(os.environ.get("TMI_PORT", 8775))  # must match `custom_port`
GAME_SPEED = float(os.environ.get("TM_GAME_SPEED", "1.0"))

TMNF_DIR = Path(os.environ["TMNF_DIR"])
TM_EXE = TMNF_DIR / "TmForever.exe"
TMLOADER_EXE = TMNF_DIR / "TMLoader.exe"

def start_tmnf():
    """Start TMLoader (which injects mods and launches TmForever) in background thread"""
    def run_in_thread():
        try:
            exe = TMLOADER_EXE if TMLOADER_EXE.exists() else TM_EXE
            print(f"Starting: {exe}")
            print(f"Exists: {exe.exists()}")

            log_path = Path("/home/wineuser/logs/tmloader.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                ["vglrun", "wine", str(exe)],
                stdout=log_file,
                stderr=log_file,
                cwd=str(exe.parent),
            )
            print(f"TMNF process started with PID: {proc.pid}")
            proc.wait()
            print(f"TMNF process exited with code: {proc.returncode}")
        except Exception as e:
            print(f"Error starting TMNF: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    return thread


def get_game_state(tm):
    """Returns position, velocity, and forward direction from the current simulation state."""
    state = tm.get_simulation_state()
    rot = state.rotation_matrix  # 3x3, columns are car's local X/Y/Z axes in world space
    return {
        "position": (state.position[0], state.position[1], state.position[2]),
        "velocity": (state.velocity[0], state.velocity[1], state.velocity[2]),
        # Third column of rotation matrix = car's local Z axis = forward direction
        "direction": (rot[0][2], rot[1][2], rot[2][2]),
    }


track = TrackInterpolator()


def get_wall_distances(game_state):
    """Cast rays from car position at my_agent.RAY_ANGLES and return distances to walls."""
    px, py, pz = game_state["position"]
    dx, dy, dz = game_state["direction"]
    car_pos = Vec3(px, py, pz)
    car_dir = Vec3(dx, dy, dz)
    return track.cast_rays(car_pos, car_dir, my_agent.RAY_ANGLES)


def send_input(tm, left=False, right=False, accelerate=False, brake=False):
    """Send driver inputs to the game."""
    tm.set_input_state(left=left, right=right, accelerate=accelerate, brake=brake)


print("=== TMRC Agent Starting ===")
print(f"TMNF_DIR: {os.environ.get('TMNF_DIR', 'NOT SET')}")
print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")

start_tmnf()

# We do NOT use a raw TCP port check here — the plugin's Render() detects every
# TCP connect as a real client, calls OnConnect(), and waits 2000ms for a
# handshake that never comes from a port-check close.
print("Waiting 20s for game and TMInterface to start...")
time.sleep(20)

print("TMInterface ready, connecting...")


def main():
    try:
        print("Connecting to TMInterface on port", PORT)
        tm = TMInterface(PORT)
        tm.register(timeout=5)
        sock = tm.sock

        # Pre-buffer SCOnConnectSync echo — plugin's WaitForResponse finds it
        # instantly on first Available check, preventing the 2000ms timeout.
        sock.sendall(struct.pack("i", MessageType.SC_ON_CONNECT_SYNC))
        print("✓ Connected")

        if GAME_SPEED != 1.0:
            tm.set_speed(GAME_SPEED)
            print(f"Set game speed to {GAME_SPEED}x")

        print("Loading map...")
        tm.execute_command('map A03-Race.Challenge.Gbx')

        print("Waiting for simulation steps...")

        step_count = 0
        while True:
            try:
                msg_type = struct.unpack("i", sock.recv(4))[0]

                if msg_type == MessageType.SC_RUN_STEP_SYNC:
                    race_time = struct.unpack("i", sock.recv(4))[0]

                    game_state = get_game_state(tm)
                    wall_distances = get_wall_distances(game_state)

                    if step_count % 100 == 0:
                        pos = game_state["position"]
                        vel = game_state["velocity"]
                        dists = " ".join(
                            f"{a:+d}°:{d:.1f}" if d is not None else f"{a:+d}°:--"
                            for a, (d, _) in zip(my_agent.RAY_ANGLES, wall_distances)
                        )
                        print(f"Step {step_count}, race_time={race_time}ms, "
                              f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), "
                              f"vel=({vel[0]:.1f},{vel[1]:.1f},{vel[2]:.1f}), "
                              f"walls=[{dists}]")
                    step_count += 1

                    inputs = my_agent.decide(game_state, wall_distances)
                    send_input(tm, **inputs)

                    sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

                elif msg_type == MessageType.SC_ON_CONNECT_SYNC:
                    sock.sendall(struct.pack("i", MessageType.SC_ON_CONNECT_SYNC))

                else:
                    sock.sendall(struct.pack("i", msg_type))

            except KeyboardInterrupt:
                print("\nShutdown requested")
                break
            except BlockingIOError:
                continue
            except struct.error as e:
                print(f"Protocol error: {e}")
                break
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                break

        tm.close()
        print("Agent shutting down gracefully")

    except Exception as e:
        print(f"ERROR: Failed to connect to TMInterface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
