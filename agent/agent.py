import os
import sys

from tminterface2 import TMInterface, MessageType
import struct

import time
import subprocess
import threading
from pathlib import Path

PORT = 8775  # must match `custom_port`

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



def wait_for_tminterface(port, timeout=60, check_interval=1):
    """Wait for TMInterface to be listening on the given port"""
    import socket
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()

            if result == 0:
                print(f"TMInterface is listening on port {port}")
                return True
        except Exception as e:
            print(f"Connection check failed: {e}")

        time.sleep(check_interval)

    print(f"Timeout waiting for TMInterface on port {port}")
    return False

print("=== TMRC Agent Starting ===")
print(f"TMNF_DIR: {os.environ.get('TMNF_DIR', 'NOT SET')}")
print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")

# Start TMNF (TMLoader auto-launches TmForever+mods via autoLaunch() in index.jsx)
start_tmnf()

# Wait for game and TMInterface to start.
# We do NOT use wait_for_tminterface (raw TCP port check) here — the plugin's
# Render() detects every TCP connect as a real client, calls OnConnect(), and
# waits 2000ms for a handshake that never comes from a port-check close.
print("Waiting 20s for game and TMInterface to start...")
time.sleep(20)

print("TMInterface ready, connecting...")

def get_game_state(tm):
    """Returns position (x,y,z) and velocity (x,y,z) from the current simulation state."""
    state = tm.get_simulation_state()
    return {
        "position": (state.position[0], state.position[1], state.position[2]),
        "velocity": (state.velocity[0], state.velocity[1], state.velocity[2]),
    }


def send_input(tm, left=False, right=False, accelerate=False, brake=False):
    """Send driver inputs to the game."""
    tm.set_input_state(left=left, right=right, accelerate=accelerate, brake=brake)


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

        # Load the map so a race starts. TMInterface searches under the user Tracks
        # folder; A01-Race is staged at Tracks/Challenges/ by the Dockerfile.
        print("Loading map...")
        tm.execute_command('map A01-Race.Challenge.Gbx')

        print("Waiting for simulation steps...")

        step_count = 0
        while True:
            # --- wait for server message ---
            try:
                msg_type = struct.unpack("i", sock.recv(4))[0]

                if msg_type == MessageType.SC_RUN_STEP_SYNC:
                    # read race time (int)
                    race_time = struct.unpack("i", sock.recv(4))[0]

                    # get position and velocity from sim state
                    game_state = get_game_state(tm)

                    if step_count % 100 == 0:
                        pos = game_state["position"]
                        vel = game_state["velocity"]
                        print(f"Step {step_count}, race_time={race_time}ms, "
                              f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), "
                              f"vel=({vel[0]:.1f},{vel[1]:.1f},{vel[2]:.1f})")
                    step_count += 1

                    # always drive forward
                    send_input(tm, accelerate=True)

                    # acknowledge step (THIS UNBLOCKS THE SIM)
                    sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

                elif msg_type == MessageType.SC_ON_CONNECT_SYNC:
                    # Shouldn't arrive here — already pre-acknowledged on connect
                    # but handle it just in case of reconnect
                    sock.sendall(struct.pack("i", MessageType.SC_ON_CONNECT_SYNC))

                else:
                    # ignore other sync events safely
                    sock.sendall(struct.pack("i", msg_type))

            except KeyboardInterrupt:
                print("\nShutdown requested")
                break
            except BlockingIOError:
                # SO_RCVTIMEO expired — no data yet, retry (map still loading etc.)
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
