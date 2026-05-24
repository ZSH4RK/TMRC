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

            proc = subprocess.Popen(
                ["wine", str(exe)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(exe.parent),  # game must run from its own directory
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

def xdotool(*args):
    """Run xdotool command with DISPLAY set"""
    subprocess.run(["xdotool"] + list(args),
                   env={**os.environ, "DISPLAY": ":99"},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def navigate_to_race():
    """Navigate TMNF menus to start A01-Race automatically.

    All coordinates are absolute screen coords on the 1024x768 Xvfb display.
    The TmForever window is at (4, 30), size 640x480, no WM decorations.
    """
    result = subprocess.run(
        ["xdotool", "search", "--name", "TrackMania Modded Forever"],
        capture_output=True, text=True,
        env={**os.environ, "DISPLAY": ":99"}
    )
    windows = [w for w in result.stdout.strip().split("\n") if w]
    if not windows:
        print("WARNING: Could not find TmForever window for navigation")
        return
    win_id = windows[0]
    print(f"Found TM window: {win_id}")

    xdotool("windowfocus", "--sync", win_id)
    time.sleep(1)

    # Click Play Solo (window coords: x=66, y=70 = screen 70,100 given window at 4,30)
    xdotool("mousemove", "--window", win_id, "66", "70")
    xdotool("click", "--window", win_id, "1")
    print("Clicked Play Solo")
    time.sleep(5)

    # Click White repeatedly until track thumbnails appear (sometimes needs multiple clicks)
    for _ in range(3):
        xdotool("mousemove", "67", "101")
        time.sleep(0.3)
        xdotool("click", "1")
        time.sleep(1)
    print("Clicked White (x3)")
    time.sleep(2)

    # Double-click A01-Race at thumbnail center (first track in White campaign)
    xdotool("mousemove", "213", "190")
    time.sleep(0.3)
    xdotool("click", "--repeat", "2", "1")
    print("Clicked A01-Race, waiting for race to load...")
    time.sleep(5)

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

# Start TMNF
start_tmnf()

# Wait for TMInterface to be ready (give it up to 60 seconds)
print("Waiting for TMInterface to be ready...")
if not wait_for_tminterface(PORT, timeout=60):
    print("ERROR: TMInterface failed to become available")
    sys.exit(1)

print("TMInterface is ready!")

# Navigate menus to start A01-Race
print("Navigating to A01-Race...")
time.sleep(8)  # Allow main menu to fully render
navigate_to_race()

def main():
    try:
        print("Connecting to TMInterface on port", PORT)
        tm = TMInterface(PORT)
        tm.register(timeout=5)
        print("✓ Agent connected to TMInterface")

        sock = tm.sock  # direct access is intentional here

        print("Race started via menu navigation; waiting for simulation steps...")

        step_count = 0
        while True:
            # --- wait for server message ---
            try:
                msg_type = struct.unpack("i", sock.recv(4))[0]

                if msg_type == MessageType.SC_RUN_STEP_SYNC:
                    # read race time (int)
                    race_time = struct.unpack("i", sock.recv(4))[0]

                    if step_count % 100 == 0:
                        print(f"Simulation step {step_count}, race_time={race_time}ms")
                    step_count += 1

                    # acknowledge step (THIS UNBLOCKS THE SIM)
                    sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

                    # always press gas
                    tm.set_input_state(
                        left=False,
                        right=False,
                        accelerate=True,
                        brake=False
                    )

                elif msg_type == MessageType.SC_ON_CONNECT_SYNC:
                    print("✓ Connection sync received")
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
