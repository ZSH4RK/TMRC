# 10. How the framework works

This page is for people who want to change or extend the framework itself.
You don't need it to use the framework.

## File map

```
TMNF-Docker/
├── docker-compose.yml          settings and how the container is run
├── docker/Dockerfile.combined  builds the image: game + framework
├── scripts/run-agent.sh        picks what to run from TM_MODE
└── TMRC/
    ├── agent/
    │   ├── env.py              TMNFEnv: the game as reset()/step(); reward; episode rules
    │   ├── track_features.py   observation, action conversion, track progress
    │   ├── tminterface2.py     low-level messages to/from the game plugin
    │   ├── tm_launch.py        starts one game
    │   ├── instance_launch.py  starts N games, each in its own Wine prefix, port and display
    │   │
    │   ├── agent.py            drive mode: one game, one decide() agent, forever
    │   ├── launcher.py         drive mode with N games (starts N agent.py processes)
    │   ├── evaluate.py         eval mode
    │   ├── verify_replay.py    verify mode
    │   ├── train_sac.py        SAC, one game
    │   ├── train_sac_vec.py    SAC, N games feeding one learner
    │   ├── sac/                SAC networks, replay buffer, background learner process
    │   ├── my_agent.py         decide() agent that drives the trained SAC network
    │   ├── compare_runs.py     leaderboard / diff of recorded runs
    │   │
    │   ├── experiments/        the recording framework (below)
    │   └── examples/           example agents and a learning script
    ├── tests/                  pytest tests for the framework (no game needed)
    ├── results/                recorded runs (shared with the host)
    └── docs/                   you are here
```

## The recording framework: `agent/experiments/`

| Module | Job |
|---|---|
| `run.py` | `Run`: creates a run folder, writes the manifest, appends episode rows and metrics, saves replays and checkpoints, writes the summary |
| `recording.py` | `RecordingEnv`: wraps a `TMNFEnv` and records every episode that passes through it |
| `replay.py` | `EpisodeRecorder` (collects steps), `Replay` (save/load, TMInterface script export), `compare_trajectories` |
| `policies.py` | `load_policy(spec)`: turns `sac:…`, `agent:…`, `forward`, `random` into objects with an `act()` method |
| `stats.py` | episode CSV columns, race ranking, summaries |
| `provenance.py` | hashes and copies of code, game files and checkpoints; environment and hardware info |
| `seeding.py` | seeds Python, NumPy and PyTorch |

### How a run is shared

- **One process, many threads** (`train_sac_vec.py`, `evaluate.py`): a single `Run`
  object, which locks around its writes.
- **Many processes** (`launcher.py` → N × `agent.py`): the launcher calls
  `Run.create()`, which sets the `TM_RUN_DIR` environment variable. The child
  processes inherit it and call `Run.attach()`, and each writes its own
  `episodes-instance-N.csv`. Only the creator writes `manifest.json` and `summary.json`.

### Step data flow

```
TMNFEnv.step_inputs(keys)
   └─ info: inputs, input_time_ms, position, velocity, direction, step_gap, agent_ms, …
RecordingEnv._record()
   └─ EpisodeRecorder.add(...)                  one step
      … on done:
RecordingEnv.end_episode()
   └─ Run.log_episode(row, replay)
         ├─ is this the best so far? (stats.rank_key)
         ├─ save replay?              (TM_SAVE_REPLAYS)
         └─ append to episodes.csv
```

## Common extensions

**A new kind of driver spec** (for example `onnx:<file>`): add a class with
`label`, `files` and `act(game_state, wall_distances, obs)` to `policies.py`,
and a branch in `load_policy()`.

**A new per-episode statistic:** compute it in `EpisodeRecorder.episode_stats()`
and add the column name to `EPISODE_FIELDS` in `stats.py`.

**A new number over time:** call `run.log_metrics("name", {...})` from anywhere
you have the run. It appears in `metrics/name.csv`.

**A new mode:** write the script (follow `evaluate.py`: create a `Run`, launch
games with `instance_launch.launch_games`, wrap envs in `RecordingEnv`, call
`run.finalize`), then add a line to `scripts/run-agent.sh`. Or skip the
mode and use `TM_MODE=script`.

**Changing what's recorded about the environment:** `env.describe()` returns
every constant that shapes an episode. It's stored in manifests and in every
replay, and `verify_replay.py` checks the important ones match.

## Tests

```bash
cd TMRC
uv run pytest
```

The tests replace the game with fakes, so they run anywhere in seconds. They
cover runs, replays, statistics, policies, the comparison tool and the
`TMNFEnv` → `RecordingEnv` hand-off. They don't test the real game connection.
For that, run an eval and verify one of its replays.

## Known limitations

- Replays are not native TrackMania `.Replay.Gbx` files. That format isn't documented.
  The `.inputs.txt` scripts haven't yet been checked in-game for exact timing.
- Only A03-Race has track data, and `track_features` loads one map per process.
- Multi-instance training isn't bit-for-bit repeatable (thread timing).
- Inside the container git isn't available, so code is identified by hash. The
  commit is only recorded if passed at build time (`TMRC_GIT_COMMIT`).
