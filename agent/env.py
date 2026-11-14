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
import time

import numpy as np

import track_features
from game.tminterface2 import MessageType, TMInterface

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
# Taken off the reward for every step that ends touching a wall
# (has_any_lateral_contact). Scraping a wall already costs speed, so progress
# reward falls anyway; this makes it cost more, to teach the car to stay off
# them. For scale: a step at 200 km/h earns ~2.8 (2.8 m of progress), so 1.0
# takes away about a third of it. 0 turns it off (the original reward).
WALL_PENALTY = float(os.environ.get("TM_WALL_PENALTY", "1.0"))


def reward_config() -> dict:
    """The reward settings. Scores, and the best-score thresholds saved with
    checkpoints, are only comparable between runs where these match."""
    return {
        "reward_per_m_advanced": REWARD_PER_M_ADVANCED,
        "reward_per_ms": REWARD_PER_MS,
        "finish_bonus": FINISH_BONUS,
        "wall_penalty": WALL_PENALTY,
    }


# Max no-op steps to wait for race_time to return to the countdown after
# give_up(), and again for the countdown to finish.
RESET_MAX_STEPS = 500

# Plugin-side no-response timeout (ms) before it drops this client.
PLUGIN_TIMEOUT_MS = 60_000

# Receive timeout (s) for the TMInterface socket once connected.
RECV_TIMEOUT_S = 120


def describe(map_file=None, game_speed=DEFAULT_GAME_SPEED) -> dict:
    """Every constant that shapes an episode, for run manifests and replay
    files. Two results are only comparable if these match."""
    return {
        "map_name": track_features.MAP_NAME,
        "map_file": map_file or f"{track_features.MAP_NAME}.Challenge.Gbx",
        "game_speed": game_speed,
        "step_ms": track_features.STEP_MS,
        "ray_angles": list(track_features.RAY_ANGLES),
        "state_dim": track_features.STATE_DIM,
        "track_length_m": track_features.TRACK_LENGTH,
        "steer_deadzone": track_features.STEER_DEADZONE,
        "throttle_deadzone": track_features.THROTTLE_DEADZONE,
        "max_episode_ms": MAX_EPISODE_MS,
        "stuck_ms": STUCK_MS,
        "min_progress_m": MIN_PROGRESS_M,
        **reward_config(),
    }


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
        # The same map track_features loads (TM_MAP_NAME): the rays and the
        # centre-line features must describe the map being driven.
        self.track = TrackInterpolator(map_name=track_features.MAP_NAME)

        self._prev_progress = 0.0
        self._prev_race_time = 0
        self._steps_without_progress = 0
        self._distance_m = 0.0
        self._stuck_ref_m = 0.0
        self._finish_time_ms = None
        # Race time of each checkpoint crossed this episode, the finish last.
        self._cp_times = []
        # Whether the car touched a wall at the latest step.
        self._wall_contact = False
        self._map_loaded = False

        # The raw state behind the latest observation, for agents that decide
        # from game state rather than the observation vector (my_agent.decide,
        # competition entries) and for recording.
        self.last_game_state = None
        self.last_wall_distances = None
        self.last_race_time = None
        # Wall-clock time the latest observation was handed back, to measure
        # how long the agent took to decide.
        self._returned_at = None

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
        self._cp_times = []
        self.last_race_time = race_time

        obs = self._build_obs(game_state)
        self._returned_at = time.perf_counter()
        return obs

    def step(self, action):
        """Step with a continuous (steer, throttle) action, as SAC produces."""
        return self.step_inputs(track_features.action_to_inputs(action))

    def step_inputs(self, inputs):
        """Step with boolean driver inputs: {left, right, accelerate, brake}.
        Any agent can drive through this, whatever its action space."""
        called_at = time.perf_counter()
        agent_ms = None if self._returned_at is None else (called_at - self._returned_at) * 1000
        input_time_ms = self._prev_race_time

        self.tm.set_input_state(**inputs)
        self.tm.sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

        game_state, race_time = self._wait_for_step()
        game_ms = (time.perf_counter() - called_at) * 1000
        obs = self._build_obs(game_state)

        progress = track_features.progress_along_track(*self._xz(game_state))
        delta = track_features.progress_delta(self._prev_progress, progress)
        dt_ms = race_time - self._prev_race_time
        reward = REWARD_PER_M_ADVANCED * delta + REWARD_PER_MS * dt_ms
        if self._wall_contact:
            reward -= WALL_PENALTY

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
            # For recording/replay: what was sent, when, and where it led.
            "inputs": dict(inputs),
            "input_time_ms": input_time_ms,
            "position": game_state["position"],
            "velocity": game_state["velocity"],
            "direction": game_state["direction"],
            "wall_contact": self._wall_contact,
            # Health: the plugin blocks the game until we answer, so race time
            # should advance exactly STEP_MS per step. Anything else means a
            # step was missed (e.g. a plugin timeout) and the run isn't the
            # experiment that was configured. Except at the finish: the game
            # reports that step at the finish time, mid-step (27980, not 28000).
            "step_gap": dt_ms != track_features.STEP_MS and self._finish_time_ms is None,
            "agent_ms": agent_ms,
            "game_ms": game_ms,
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
        if done:
            # Splits: where the time went. info["position"] is where the
            # episode ended: at the finish, or where it got stuck or timed out.
            info["checkpoint_times_ms"] = list(self._cp_times)

        self.last_race_time = race_time
        self._returned_at = time.perf_counter()
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
                # Kept out of game_state: that's the competition's decide()
                # interface, which doesn't include it.
                self._wall_contact = bool(state.scene_mobil.has_any_lateral_contact)
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
                if msg_type == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    finished = current == target and target > 0
                    if finished:
                        # Crossed the finish. Must be called before acking this
                        # message: if the game goes on to end the race it stops
                        # stepping and ignores give_up(), so reset() would hang.
                        self.tm.prevent_simulation_finish()
                    if finished or current == len(self._cp_times) + 1:
                        # From the state, not the next step: that's up to a step
                        # late. The game's own checkpoint times aren't filled in
                        # yet at this point. (A restart's count going back to 0
                        # isn't a checkpoint; reset() clears the list.)
                        cp_time = self.tm.get_simulation_state().race_time
                        if current == len(self._cp_times) + 1:
                            self._cp_times.append(cp_time)
                        if finished:
                            self._finish_time_ms = cp_time
                sock.sendall(struct.pack("i", msg_type))
            else:
                sock.sendall(struct.pack("i", msg_type))

    def _build_obs(self, game_state):
        px, py, pz = game_state["position"]
        dx, dy, dz = game_state["direction"]
        wall_distances = self.track.cast_rays(Vec3(px, py, pz), Vec3(dx, dy, dz), track_features.RAY_ANGLES)
        self.last_game_state = game_state
        self.last_wall_distances = wall_distances
        return np.array(track_features.build_state(game_state, wall_distances), dtype=np.float32)

    @staticmethod
    def _xz(game_state):
        x, _y, z = game_state["position"]
        return x, z
