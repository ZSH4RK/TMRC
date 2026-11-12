"""Recording, replay and evaluation framework for every TMRC entry point.

Every way of running the game (plain driving agents via launcher.py/agent.py,
the SAC trainers, evaluate.py and verify_replay.py) records into the same
layout, so any two results can be compared and any result can be reproduced:

    results/<run_id>/
        manifest.json      what ran: code hashes, config, TM_* env, seeds,
                           map/track/plugin hashes, hardware
        code/              snapshot of the agent source, plugins and track code
        episodes*.csv      one row per episode (one file per writing process)
        metrics/*.csv      time series: perf, learner losses, ...
        replays/*.npz      recorded episodes (inputs + trajectory), each with
                           a matching TMInterface input script (*.inputs.txt)
        checkpoints/       model weights the run produced or started from
        summary.json       written on finish: aggregate results + status

Modules:
    run         Run: creates/attaches to a run directory and writes into it
    replay      EpisodeRecorder / Replay: per-step recording, save/load,
                TMInterface script export, trajectory comparison
    recording   RecordingEnv: wraps TMNFEnv so any agent driving it is recorded
    policies    load_policy(): one interface for SAC checkpoints, agent files
                (the competition's decide() interface) and baselines
    provenance  hashing / environment capture used by the manifest
    seeding     seed_everything()
    stats       episode summaries and the racing rank used everywhere
"""

from .policies import load_policy
from .recording import RecordingEnv
from .replay import EpisodeRecorder, Replay
from .run import Run
from .seeding import seed_everything

__all__ = ["EpisodeRecorder", "RecordingEnv", "Replay", "Run", "load_policy", "seed_everything"]
