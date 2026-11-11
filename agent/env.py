"""Gym-classic-style environment wrapping TMInterface, for SAC training.

Bridges the game-driven protocol (the game pushes SC_RUN_STEP_SYNC messages
and blocks for a response) to the agent-driven step()/reset() interface SAC
expects. Follows the shape sketched in ../Plan.md's "Phase 3 — TMNFEnv",
adapted to:
  - a continuous 2D action space (steer, throttle) instead of a discrete
    action list, since SAC needs a continuous action space to reparameterize
    through;
  - gym-classic's `step() -> (obs, reward, done, info)` 4-tuple rather than
    gymnasium's 5-tuple, to match Soft-Actor-Critic/main_sac.py's calling
    convention (and sac.agent.SACAgent.remember's signature) directly.
"""

import os
import socket
import struct
import sys

import numpy as np

import track_features
from tminterface2 import MessageType, TMInterface

sys.path.insert(0, str(track_features.TRACKS_DIR))
from interpolate_track import TrackInterpolator, Vec3

# Simulation speed multiplier passed to TMInterface's C_SET_SPEED. race_time
# below is in-game/simulated ms (the same clock the in-game timer shows) —
# it's unaffected by this; a higher multiplier just means more simulated ms
# elapse per wall-clock second, so episodes finish faster in real time.
# Matches game-data/TMInterface/builtin.txt's sim_speed default.
DEFAULT_GAME_SPEED = float(os.environ.get("TM_GAME_SPEED", "5.0"))

# Episode ends ("timeout") if the map isn't finished within this long, in ms
# of in-game race time — i.e. 2 minutes on the in-game clock, not wall-clock
# time (which is ~1/DEFAULT_GAME_SPEED shorter). Must leave a still-learning
# agent time to reach the finish, or it can never earn FINISH_BONUS.
MAX_EPISODE_MS = 120_000
# Episode ends early ("stuck") if the car hasn't advanced along the
# centerline by at least MIN_PROGRESS_M within this many consecutive steps —
# catches wall-stuck / stopped-facing-backward cases without waiting for the
# full timeout.
STUCK_MS = 3000
STUCK_STEPS = STUCK_MS // track_features.STEP_MS
MIN_PROGRESS_M = 1.0

# Reward shaping. Not the same tuned values as config_files/config_copy.py's
# (unused) reward_per_m_advanced_along_centerline / constant_reward_per_ms —
# those were tuned for a different agent/observation setup — but the same
# idea: reward forward progress along the centerline, penalize time taken,
# and give a one-off bonus for actually finishing.
REWARD_PER_M_ADVANCED = 1.0
REWARD_PER_MS = -6 / 5000
FINISH_BONUS = 100.0


# Max no-op steps to wait for race_time to return to the countdown after
# give_up(), and again for the countdown to finish.
RESET_MAX_STEPS = 500

# Plugin-side no-response timeout (ms) before it drops this client.
PLUGIN_TIMEOUT_MS = 60_000

# Receive timeout (s) for the TMInterface socket once connected.
RECV_TIMEOUT_S = 120


class TMNFEnv:
    """Wraps a single TMInterface connection as a step()/reset() RL environment."""

    def __init__(
        self,
        port: int,
        map_file: str = f"{track_features.MAP_NAME}.Challenge.Gbx",
        game_speed: float = DEFAULT_GAME_SPEED,
    ):
        self.port = port
        self.map_file = map_file
        self.game_speed = game_speed
        self.tm = None
        self.track = TrackInterpolator()

        self._prev_progress = 0.0
        self._prev_race_time = 0
        self._steps_without_progress = 0
        self._distance_m = 0.0
        self._stuck_ref_m = 0.0
        self._finish_time_ms = None
        self._map_loaded = False

    def connect(self):
        """Open the TCP connection to TMInterface. Call once before the first reset()."""
        self.tm = TMInterface(self.port)
        self.tm.register(timeout=5)
        # Pre-buffer the SCOnConnectSync echo — the plugin's WaitForResponse
        # finds it instantly on the first Available check, avoiding a
        # 2000ms handshake timeout (see agent.py's main() / TMNF-Docker
        # project memory notes on this).
        self.tm.sock.sendall(struct.pack("i", MessageType.SC_ON_CONNECT_SYNC))
        self.tm.set_speed(self.game_speed)
        self.tm.set_on_step_period(track_features.STEP_MS)
        # Python_Link.as drops a client that doesn't answer a sync message
        # within RESPONSE_TIMEOUT (default 2000ms). Lock-step vector training
        # makes each instance wait on the slowest one plus a learn() step, so
        # raise it well above that.
        self.tm.set_timeout(PLUGIN_TIMEOUT_MS)
        # register(timeout=5) leaves a 5s recv timeout on the socket, which a
        # map load (several seconds, longer with N games sharing the CPU) exceeds.
        recv_timeout = struct.pack("ll", RECV_TIMEOUT_S, 0)
        self.tm.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, recv_timeout)

    def reset(self):
        if self.tm is None:
            self.connect()

        if not self._map_loaded:
            self.tm.execute_command(f"map {self.map_file}")
            self._map_loaded = True
        else:
            # Restart the run in place (like the in-game restart key), as
            # agent.py does. The plugin is blocked in WaitForResponse for the
            # step that ended the last episode, so it must be acked or both
            # sides wait on each other until the plugin drops the client.
            self.tm.give_up()
            self.tm.sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

        game_state, race_time = self._wait_for_step()
        # After give_up the first few steps can still carry the old race_time;
        # step (no-op) until the clock is back at/below 0 (the pre-start countdown).
        for _ in range(RESET_MAX_STEPS):
            if race_time <= 0:
                break
            self.tm.sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))
            game_state, race_time = self._wait_for_step()
        # The countdown can't be driven, so inputs there teach the agent nothing:
        # step through it here to keep it out of the episode and replay buffer.
        for _ in range(RESET_MAX_STEPS):
            if race_time >= 0:
                break
            self.tm.sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))
            game_state, race_time = self._wait_for_step()

        self._prev_progress = track_features.progress_along_track(*self._xz(game_state))
        self._prev_race_time = race_time
        self._steps_without_progress = 0
        self._distance_m = 0.0
        self._stuck_ref_m = 0.0
        self._finish_time_ms = None

        return self._build_obs(game_state)

    def step(self, action):
        inputs = track_features.action_to_inputs(action)
        self.tm.set_input_state(**inputs)
        self.tm.sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

        game_state, race_time = self._wait_for_step()
        obs = self._build_obs(game_state)

        progress = track_features.progress_along_track(*self._xz(game_state))
        delta = track_features.progress_delta(self._prev_progress, progress)
        dt_ms = race_time - self._prev_race_time
        reward = REWARD_PER_M_ADVANCED * delta + REWARD_PER_MS * dt_ms

        self._distance_m += delta
        if self._distance_m >= self._stuck_ref_m + MIN_PROGRESS_M:
            self._stuck_ref_m = self._distance_m
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1
        self._prev_progress = progress
        self._prev_race_time = race_time

        done = False
        # distance_m: metres driven along the centreline this episode (net of
        # reversing), so a run that doesn't finish still has a comparable score.
        info = {
            "distance_m": self._distance_m,
            "track_pct": 100.0 * self._distance_m / track_features.TRACK_LENGTH,
            "race_time_ms": race_time,
        }

        if self._finish_time_ms is not None:
            reward += FINISH_BONUS
            done = True
            info["done_reason"] = "finished"
            info["race_time_ms"] = self._finish_time_ms
        elif race_time >= MAX_EPISODE_MS:
            done = True
            info["done_reason"] = "timeout"
        elif self._steps_without_progress >= STUCK_STEPS:
            done = True
            info["done_reason"] = "stuck"

        return obs, reward, done, info

    def close(self):
        if self.tm is not None:
            self.tm.close()
            self.tm = None

    def _wait_for_step(self):
        """Drain messages until SC_RUN_STEP_SYNC, acking everything else in
        between — mirrors agent.py's main loop dispatch."""
        sock = self.tm.sock
        while True:
            header = sock.recv(4)
            if len(header) < 4:
                raise ConnectionError(f"TMInterface on port {self.port} closed the connection")
            msg_type = struct.unpack("i", header)[0]

            if msg_type == MessageType.SC_RUN_STEP_SYNC:
                race_time = struct.unpack("i", sock.recv(4))[0]
                state = self.tm.get_simulation_state()
                rot = state.rotation_matrix  # 3x3, columns are car's local X/Y/Z axes in world space
                game_state = {
                    "position": (state.position[0], state.position[1], state.position[2]),
                    "velocity": (state.velocity[0], state.velocity[1], state.velocity[2]),
                    # Third column of rotation matrix = car's local Z axis = forward direction
                    "direction": (rot[0][2], rot[1][2], rot[2][2]),
                }
                return game_state, race_time
            elif msg_type in (MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                # Python_Link.as writes msg_type + current(int32) + target(int32)
                # before WaitForResponse — both extra ints must be drained here,
                # or the next recv() misreads "current" as the next message-type
                # header and desyncs the whole protocol stream.
                current, target = struct.unpack("ii", sock.recv(8, socket.MSG_WAITALL))
                if msg_type == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC and current == target and target > 0:
                    # Crossed the finish. Must be called before acking this
                    # message: if the game goes on to end the race it stops
                    # stepping and ignores give_up(), so reset() would hang.
                    self.tm.prevent_simulation_finish()
                    # From the state, not the next step: that's up to a step
                    # late. The checkpoint times aren't filled in yet at this point.
                    self._finish_time_ms = self.tm.get_simulation_state().race_time
                sock.sendall(struct.pack("i", msg_type))
            else:
                sock.sendall(struct.pack("i", msg_type))

    def _build_obs(self, game_state):
        px, py, pz = game_state["position"]
        dx, dy, dz = game_state["direction"]
        wall_distances = self.track.cast_rays(Vec3(px, py, pz), Vec3(dx, dy, dz), track_features.RAY_ANGLES)
        return np.array(track_features.build_state(game_state, wall_distances), dtype=np.float32)

    @staticmethod
    def _xz(game_state):
        x, _y, z = game_state["position"]
        return x, z
