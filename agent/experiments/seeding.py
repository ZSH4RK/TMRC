"""Seeding for repeatable runs.

TrackMania's physics are deterministic for a given input sequence, so a
deterministic policy repeats exactly with or without a seed. Seeds matter for
anything sampled: SAC's exploration noise, network initialisation, random
baselines. Multi-instance training still isn't bit-for-bit repeatable: the game
threads interleave differently every run. The seed makes it as repeatable as
it can be, and is recorded either way.
"""

import os
import random
import secrets
from typing import Optional

import numpy as np


def resolve_seed(seed: Optional[int] = None) -> int:
    """The seed to use: the argument, else TM_SEED, else a fresh random one
    (recorded in the manifest, so the run can still be repeated)."""
    if seed is not None:
        return int(seed)
    env_seed = os.environ.get("TM_SEED", "").strip()
    if env_seed:
        return int(env_seed)
    return secrets.randbelow(2**31 - 1)


def seed_everything(seed: Optional[int] = None) -> int:
    seed = resolve_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
