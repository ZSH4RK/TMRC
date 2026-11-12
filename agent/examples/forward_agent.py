"""The smallest possible agent: hold accelerate, never steer.

Every agent is a Python file with a decide() function. The framework calls it
once per step (every TM_STEP_MS of game time, 50ms by default) and sends the
keys it returns to the game.

Try it:
    TM_MODE=eval TM_EVAL_AGENT=agent:examples/forward_agent.py docker compose up --build
"""


def decide(game_state, wall_distances):
    # game_state: {"position": (x, y, z), "velocity": (vx, vy, vz), "direction": (dx, dy, dz)}
    # wall_distances: one (distance_in_metres_or_None, "left"/"right"/None) per ray
    return {"left": False, "right": False, "accelerate": True, "brake": False}
