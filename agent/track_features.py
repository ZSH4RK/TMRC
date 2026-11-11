"""Shared observation/action interface between training and inference.

Both the SAC training loop (`env.py`, `train_sac.py`) and the inference agent
(`my_agent.py`) need to build the exact same state vector from raw game state,
and turn continuous actions back into the boolean inputs TMInterface expects.
Keeping that logic in one place guarantees the network sees the same features
at train and inference time.
"""

import csv
import math
import os
from pathlib import Path

import numpy as np

# Angles (degrees) at which rays are cast from the car to measure wall distances.
# 0 = straight ahead, negative = left, positive = right.
RAY_ANGLES = [-90, -45, -10, 0, 10, 45, 90]

# Wall distances (len(RAY_ANGLES)) + vx, vz, dx, dz, angle_to_centreline,
# vel_angle_to_centreline, dist_to_centreline.
STATE_DIM = len(RAY_ANGLES) + 7

MAP_NAME = os.environ.get("TM_MAP_NAME", "A03-Race")

# Game time (ms) between agent decisions, multiple of the 10ms physics tick.
# Lives here because training and inference must act at the same rate: inputs
# are held between decisions, so a policy trained at one rate drives
# differently at another.
STEP_MS = int(os.environ.get("TM_STEP_MS", "50"))

# Scales that bring every observation to roughly [-1, 1], so no single feature
# dominates the networks' inputs. Wall distance saturates: only nearby walls
# affect driving, and rays can otherwise reach 500m.
MAX_WALL_DIST_M = 100.0
MAX_SPEED_MS = 100.0
MAX_CENTRELINE_DIST_M = 20.0

# In Docker, docker-compose.yml bind-mounts ../Tracks to agent/../Tracks.
# On the host, Tracks is a sibling of TMNF-Docker instead of TMRC, one level
# higher. Auto-detect between the two, or set TRACKS_DIR to override.
if "TRACKS_DIR" in os.environ:
    _TRACKS_DIR = Path(os.environ["TRACKS_DIR"])
else:
    _container_tracks_dir = Path(__file__).parent.parent / "Tracks"
    _host_tracks_dir = Path(__file__).parent.parent.parent.parent / "Tracks"
    _TRACKS_DIR = _container_tracks_dir if _container_tracks_dir.exists() else _host_tracks_dir

# Public alias: other modules (agent.py, env.py) need this same directory to
# find interpolate_track.py, which lives alongside the per-map CSVs.
TRACKS_DIR = _TRACKS_DIR

_TRACKS_CSV = _TRACKS_DIR / MAP_NAME / "points-offset.csv"

_mid_points = []
with open(_TRACKS_CSV, newline="") as _f:
    for _row in csv.DictReader(_f):
        if _row["type"] == "mid":
            _mid_points.append((float(_row["x"]), float(_row["y"]), float(_row["z"])))

_mid_xz = np.array([[p[0], p[2]] for p in _mid_points])

# Cumulative arc length (meters) along the centerline, one entry per
# _mid_points index. Turns "nearest point index" into a scalar
# progress-along-track value for reward shaping.
_segment_lengths = np.sqrt(np.sum(np.diff(_mid_xz, axis=0) ** 2, axis=1))
_cum_dist = np.concatenate([[0.0], np.cumsum(_segment_lengths)])
TRACK_LENGTH = float(_cum_dist[-1])

_seg_starts = _mid_xz[:-1]
_seg_vectors = np.diff(_mid_xz, axis=0)
_seg_lengths_sq = np.maximum(_segment_lengths ** 2, 1e-12)


def nearest_midpoint(car_x, car_z, lookahead=30.0):
    """Return (index, nearest, lookahead) where nearest and lookahead are (x, y, z) tuples.

    lookahead is the midpoint `lookahead` world units ahead along the centerline.
    """

    diffs = _mid_xz - np.array([car_x, car_z])
    idx = int(np.argmin(np.sum(diffs ** 2, axis=1)))

    dist = 0.0
    n = len(_mid_points)
    ahead_idx = idx
    while dist < lookahead:
        next_idx = (ahead_idx + 1) % n
        p0, p1 = _mid_points[ahead_idx], _mid_points[next_idx]
        dist += ((p1[0] - p0[0]) ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5
        ahead_idx = next_idx

    return idx, _mid_points[idx], _mid_points[ahead_idx]


def progress_along_track(car_x, car_z):
    """Cumulative distance (meters) along the centerline at the car's projection
    onto its nearest centerline segment.

    Projecting rather than snapping to the nearest point keeps progress (and
    so the reward) continuous; the points are ~2.5m apart, which a slow car
    takes many steps to cross.

    Monotonic increasing within a lap; wraps back to ~0 after the start/finish
    line. Use `progress_delta` to turn two readings into a signed advance that
    handles the wraparound correctly.
    """
    to_car = np.array([car_x, car_z]) - _seg_starts
    t = np.clip(np.sum(to_car * _seg_vectors, axis=1) / _seg_lengths_sq, 0.0, 1.0)
    offsets = to_car - t[:, None] * _seg_vectors
    idx = int(np.argmin(np.sum(offsets ** 2, axis=1)))
    return float(_cum_dist[idx] + t[idx] * _segment_lengths[idx])


def progress_delta(prev, curr):
    """Signed distance advanced along the centerline from `prev` to `curr`.

    Corrects for the wraparound at the start/finish line: if the naive delta
    is more than half the track length, the car crossed the line rather than
    driven backward almost an entire lap.
    """
    delta = curr - prev
    if delta > TRACK_LENGTH / 2:
        delta -= TRACK_LENGTH
    elif delta < -TRACK_LENGTH / 2:
        delta += TRACK_LENGTH
    return delta


def build_state(game_state, wall_distances):
    """Build the flat observation vector fed to the SAC actor.

    Args:
        game_state: dict with "position", "velocity", "direction" (see
            `agent.get_game_state`).
        wall_distances: list of (distance, wall) tuples, one per RAY_ANGLE.
            distance is None when no wall was hit within range.

    Returns:
        list[float] of length STATE_DIM.
    """
    x, y, z = game_state["position"]
    vx, vy, vz = game_state["velocity"]
    dx, dy, dz = game_state["direction"]

    _, mid_point, mid_point_ahead = nearest_midpoint(x, z, lookahead=10)

    # Angle (radians) between the velocity vector and the centreline direction,
    # both measured in the world x-z plane. 0 = moving parallel to the track,
    # positive = drifting toward the right of the centreline direction.
    centreline_angle = math.atan2(mid_point_ahead[2] - mid_point[2], mid_point_ahead[0] - mid_point[0])
    direction_angle = math.atan2(dz, dx)
    angle_to_centreline = (direction_angle - centreline_angle + math.pi) % (2 * math.pi) - math.pi
    angle_to_centreline_deg = math.degrees(angle_to_centreline)

    velocity_angle = math.atan2(vz, vx)
    vel_angle_to_centreline = (velocity_angle - centreline_angle + math.pi) % (2 * math.pi) - math.pi
    vel_angle_to_centreline_deg = math.degrees(vel_angle_to_centreline)

    dist_to_centreline = math.sqrt((x - mid_point[0]) ** 2 + (z - mid_point[2]) ** 2)

    # No hit within range means the wall is far away, the same as a saturated distance.
    wall_dists = [1.0 if d is None else min(d / MAX_WALL_DIST_M, 1.0) for d, _wall in wall_distances]
    # Wall distances (7), velocity x/z, direction x/z, angle to centre, vel angle to centre, distance to centre
    return wall_dists + [
        vx / MAX_SPEED_MS,
        vz / MAX_SPEED_MS,
        dx,
        dz,
        angle_to_centreline_deg / 180.0,
        vel_angle_to_centreline_deg / 180.0,
        dist_to_centreline / MAX_CENTRELINE_DIST_M,
    ]


# Deadzone (same units as the actor's tanh-squashed output, [-1, 1]) below
# which steer/throttle are treated as neutral. Avoids the car chattering
# between left/right or accelerate/brake when the actor outputs a value near
# zero — TMInterface only accepts boolean inputs, there is no analog steering.
STEER_DEADZONE = 0.1
THROTTLE_DEADZONE = 0.1


def action_to_inputs(action):
    """Map a continuous 2D action (steer, throttle), both in [-1, 1], to the
    boolean driver inputs TMInterface's C_SET_INPUT_STATE expects.

    steer:    -1 = full left   ... +1 = full right
    throttle: -1 = full brake  ... +1 = full accelerate
    """
    steer, throttle = float(action[0]), float(action[1])
    return {
        "left": steer < -STEER_DEADZONE,
        "right": steer > STEER_DEADZONE,
        "accelerate": throttle > THROTTLE_DEADZONE,
        "brake": throttle < -THROTTLE_DEADZONE,
    }
