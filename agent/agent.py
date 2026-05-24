import os
import sys

from tminterface2 import TMInterface, MessageType
import struct

import time
import subprocess
import threading
from pathlib import Path

PORT = 8775  # must match `custom_port`

TM_EXE = Path(os.environ["TMNF_DIR"]) / "TMLoader.exe"

def start_tmnf():
    """Start TMNF in background thread"""
    def run_in_thread():
        try:
            print(f"Starting TMNF: {TM_EXE}")
            print(f"TMNF_DIR exists: {TM_EXE.parent.exists()}")
            print(f"TMLoader.exe exists: {TM_EXE.exists()}")

            proc = subprocess.Popen(
                ["wine", str(TM_EXE)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"TMNF process started with PID: {proc.pid}")

            # Read output in the thread to avoid blocking
            while proc.poll() is None:
                time.sleep(1)

            stdout, stderr = proc.communicate(timeout=5)
            if stdout:
                print(f"TMNF stdout: {stdout[:500]}")
            if stderr:
                print(f"TMNF stderr: {stderr[:500]}")
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

# Start TMNF
start_tmnf()

# Wait for TMInterface to be ready (give it up to 60 seconds)
print("Waiting for TMInterface to be ready...")
if not wait_for_tminterface(PORT, timeout=60):
    print("ERROR: TMInterface failed to become available")
    sys.exit(1)

print("TMInterface is ready!")

def main():
    try:
        print("Connecting to TMInterface on port", PORT)
        tm = TMInterface(PORT)
        tm.register(timeout=5)
        print("✓ Agent connected to TMInterface")

        sock = tm.sock  # direct access is intentional here

        print("Loading map A01-Race...")
        tm.execute_command('map r"A01-Race.Challenge.Gbx"')
        print("✓ Map load command sent")

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
