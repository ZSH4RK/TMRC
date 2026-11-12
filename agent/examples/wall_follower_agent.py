"""A hand-written (rule-based) agent: no learning, just if-statements.

Idea: look at the rays, find how close the nearest left wall and the nearest
right wall are, and steer away from whichever is closer. Slow down if a wall
is close straight ahead.

It won't win races, but it's a good baseline. Any learned agent should beat
it, and it shows how to read the inputs. Change the numbers below and
re-evaluate to see what they do.

Try it:
    TM_MODE=eval TM_EVAL_AGENT=agent:examples/wall_follower_agent.py docker compose up --build
"""

import math

# How much closer one side must be before we steer (metres). Stops the car
# twitching left-right when it's roughly centred.
STEER_MARGIN_M = 2.0
# Brake if the wall straight ahead is closer than this, and we're fast.
BRAKE_DISTANCE_M = 25.0
BRAKE_SPEED_KMH = 120.0
# Ray readings with no hit (None) count as this far away.
FAR_M = 500.0


def nearest_wall(wall_distances, side):
    """Closest hit on the given wall ("left" or "right") across all rays.

    Each ray says which wall it hit. Using that label means we don't need to
    know which ray angle points which way.
    """
    hits = [d for d, wall in wall_distances if wall == side and d is not None]
    return min(hits) if hits else FAR_M


def speed_kmh(game_state):
    vx, vy, vz = game_state["velocity"]
    return math.sqrt(vx * vx + vy * vy + vz * vz) * 3.6


def decide(game_state, wall_distances):
    left = nearest_wall(wall_distances, "left")
    right = nearest_wall(wall_distances, "right")

    # The rays are in angle order, so the middle one (angle 0) looks straight ahead.
    ahead, _wall = wall_distances[len(wall_distances) // 2]
    if ahead is None:
        ahead = FAR_M
    too_fast = ahead < BRAKE_DISTANCE_M and speed_kmh(game_state) > BRAKE_SPEED_KMH

    return {
        "left": right + STEER_MARGIN_M < left,    # right wall closer -> steer left
        "right": left + STEER_MARGIN_M < right,   # left wall closer -> steer right
        "accelerate": not too_fast,
        "brake": too_fast,
    }
