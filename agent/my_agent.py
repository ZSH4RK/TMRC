"""
Edit this file to customize the agent's behavior.

`decide` is called once per simulation step. Return a dict with the
inputs you want to send to the car.
"""

import csv
from pathlib import Path

import numpy as np

# Angles (degrees) at which rays are cast from the car to measure wall distances.
# 0 = straight ahead, negative = left, positive = right.
RAY_ANGLES = [-90, -45, -10, 0, 10, 45, 90]

_TRACKS_CSV = Path(__file__).parent.parent / "Tracks" / "A03-Race" / "points-offset.csv"

_mid_points = []
with open(_TRACKS_CSV, newline="") as _f:
    for _row in csv.DictReader(_f):
        if _row["type"] == "mid":
            _mid_points.append((float(_row["x"]), float(_row["y"]), float(_row["z"])))

_mid_xz = np.array([[p[0], p[2]] for p in _mid_points])


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


def decide(game_state, wall_distances):
    """
    Called every simulation step. Return a dict of driver inputs.

    Args:
        game_state: dict with keys:
            "position"  – (x, y, z) world position
            "velocity"  – (vx, vy, vz) world velocity
            "direction" – (dx, dy, dz) forward unit vector
        wall_distances: list of (distance, hit_point) tuples, one per RAY_ANGLE.
            distance is None when no wall was hit within range.

    Returns:
        dict with boolean keys: "left", "right", "accelerate", "brake"
    """

    print(game_state)
    print(wall_distances)
    return dict(left=False, right=False, accelerate=True, brake=False)
