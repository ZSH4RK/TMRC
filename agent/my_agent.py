"""
Edit this file to customize the agent's behavior.

`decide` is called once per simulation step. Return a dict with the
inputs you want to send to the car.

Driving policy is the SAC actor trained by train_sac.py (see sac/agent.py /
sac/networks.py, ported from Soft-Actor-Critic/). If no trained checkpoint
exists yet under tmp/sac/, this drives with an untrained (randomly
initialized) network — same placeholder behavior as before training exists.
"""

from pathlib import Path

import torch

import track_features
from sac.networks import ActorNetwork

# Re-exported for agent.py, which casts rays at these angles before calling
# decide().
RAY_ANGLES = track_features.RAY_ANGLES

_CHKPT_DIR = str(Path(__file__).parent / "tmp" / "sac")

# Inference-only actor. Architecture and chkpt_dir must match how train_sac.py
# constructs its SACAgent, so the saved weights actually line up.
model = ActorNetwork(
    alpha=0.0003,
    input_dims=[track_features.STATE_DIM],
    max_action=[1.0, 1.0],
    n_actions=2,
    chkpt_dir=_CHKPT_DIR,
)
model.eval()

try:
    model.load_checkpoint()
    print(f"Loaded trained SAC actor from {model.checkpoint_file}")
except FileNotFoundError:
    print(
        f"No trained SAC actor checkpoint at {model.checkpoint_file} — "
        "driving with an untrained (random) policy. Run train_sac.py first."
    )


def decide(game_state, wall_distances):
    """
    Called every simulation step. Return a dict of driver inputs.

    Args:
        game_state: dict with keys:
            "position"  – (x, y, z) world position
            "velocity"  – (vx, vy, vz) world velocity
            "direction" – (dx, dy, dz) forward unit vector
        wall_distances: list of (distance, wall)) tuples, one per RAY_ANGLE.
            distance is None when no wall was hit within range.

    Returns:
        dict with boolean keys: "left", "right", "accelerate", "brake"
    """

    state = track_features.build_state(game_state, wall_distances)
    state_tensor = torch.tensor([state], dtype=torch.float32).to(model.device)

    with torch.no_grad():
        # Mean action rather than a sampled one: deterministic driving at
        # inference time, no exploration noise.
        action = model.deterministic_action(state_tensor)[0]

    return track_features.action_to_inputs(action.cpu().numpy())
