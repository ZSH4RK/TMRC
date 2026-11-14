"""SAC training settings, read from TM_ environment variables.

Both trainers build their SACAgent from these, and record describe() in the
run's manifest, so a run says which hyperparameters produced it.
"""

import os


def _bool(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# Transitions the replay buffer holds. At 50 ms steps an A03 episode is a few
# hundred transitions, so 1M keeps well over a thousand episodes. A buffer that
# only holds the last ~150 lets the critic forget the mistakes the policy has
# stopped making, and the policy then drifts back into them.
REPLAY_SIZE = int(float(os.environ.get("TM_REPLAY_SIZE", "1000000")))
# Adam learning rate for the actor, critics and value networks.
LR = float(os.environ.get("TM_LR", "0.0003"))
# Updates over which the learning rate falls linearly to LR * LR_FINAL, then
# stays there. 0 keeps it constant.
LR_DECAY_UPDATES = int(float(os.environ.get("TM_LR_DECAY_UPDATES", "0")))
LR_FINAL = float(os.environ.get("TM_LR_FINAL", "0.1"))
# Tune the entropy temperature so the policy's entropy tracks TARGET_ENTROPY,
# instead of a fixed temperature. Off reproduces the original SAC exactly.
AUTO_ENTROPY = _bool("TM_AUTO_ENTROPY", "true")
# Default -n_actions, the usual SAC choice (-2 for steer + throttle).
TARGET_ENTROPY = os.environ.get("TM_TARGET_ENTROPY")


def agent_kwargs(n_actions: int) -> dict:
    """Keyword arguments for SACAgent (except the replay buffer)."""
    return {
        "alpha": LR,
        "beta": LR,
        "lr_decay_updates": LR_DECAY_UPDATES,
        "lr_final": LR_FINAL,
        "auto_entropy": AUTO_ENTROPY,
        "target_entropy": float(TARGET_ENTROPY) if TARGET_ENTROPY else -float(n_actions),
    }


def describe(n_actions: int) -> dict:
    """The settings as recorded in a run's manifest."""
    return {"replay_size": REPLAY_SIZE, **agent_kwargs(n_actions)}
