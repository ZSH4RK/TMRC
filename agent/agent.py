import os
import socket
import sys

from tminterface2 import TMInterface, MessageType
import struct

import time
from pathlib import Path

import my_agent
import track_features
from tm_launch import start_tmnf

sys.path.insert(0, str(track_features.TRACKS_DIR))
from interpolate_track import TrackInterpolator, Vec3

# TMI_PORT is set per-instance by launcher.py when TM_NUM_INSTANCES > 1
# (instance i uses TM_BASE_PORT + i); must match this instance's `custom_port`.
PORT = int(os.environ.get("TMI_PORT", 8775))
GAME_SPEED = float(os.environ.get("TM_GAME_SPEED", "5.0"))
# Reload the map after this long, in ms of in-game race time, so a single
# run driving forever (e.g. stuck against a wall) doesn't sit there forever.
RUN_CUTOFF_MS = 120_000


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

        print("Loading map...")
        # Must match track_features.MAP_NAME (A03-Race) — that's the only map
        # with wall/centerline CSV data under Tracks/, which my_agent.decide()
        # needs to build its state vector.
        tm.execute_command(f"map {track_features.MAP_NAME}.Challenge.Gbx")
        tm.set_speed(GAME_SPEED)
        print(f"Set game speed to {GAME_SPEED}x")
        tm.set_on_step_period(track_features.STEP_MS)

        print("Waiting for simulation steps...")

        step_count = 0
        cutoff_triggered = False  # armed once per over-cutoff period, so a
        # slow-to-reset race_time across a few consecutive steps doesn't
        # trigger give_up() repeatedly mid-respawn.
        while True:
            try:
                msg_type = struct.unpack("i", sock.recv(4))[0]

                if msg_type == MessageType.SC_RUN_STEP_SYNC:
                    race_time = struct.unpack("i", sock.recv(4))[0]

                    if race_time >= RUN_CUTOFF_MS:
                        if not cutoff_triggered:
                            print(f"Run cutoff reached ({race_time}ms >= {RUN_CUTOFF_MS}ms), giving up run...")
                            # give_up() restarts the current run (like the
                            # in-game restart key) without reloading the
                            # whole map — much faster, and doesn't interrupt
                            # a map load like re-issuing execute_command("map
                            # ...") every step would.
                            tm.give_up()
                            cutoff_triggered = True
                            step_count = 0
                        sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))
                        continue
                    else:
                        cutoff_triggered = False

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

                elif msg_type in (MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                    # Python_Link.as writes msg_type + current(int32) + target(int32)
                    # before WaitForResponse — both extra ints must be drained here,
                    # or the next recv() misreads "current" as the next message-type
                    # header and desyncs the whole protocol stream.
                    current, target = struct.unpack("ii", sock.recv(8, socket.MSG_WAITALL))
                    if msg_type == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC and current == target and target > 0:
                        # Crossed the finish. Must be called before acking this
                        # message, or the game ends the race and ignores give_up().
                        print("Finished, restarting run...")
                        tm.prevent_simulation_finish()
                        tm.give_up()
                        step_count = 0
                    sock.sendall(struct.pack("i", msg_type))

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
