# 6. Results and replays

Everything you run is recorded in a folder under `TMRC/results/`. This page
explains what's in it and how to use it.

## A run folder

Names look like `20261112-153000-eval-my-label`: date, time, what kind of run
it was, and the label you gave it with `TM_RUN_NAME`.

```
results/20261112-153000-eval-my-label/
├── manifest.json      what ran: code, settings, seed, game files, computer
├── summary.json       how it went, in one place (written at the end)
├── episodes.csv       one row per episode
├── replays/           recordings of the best episodes
├── metrics/           numbers over time (for training runs)
├── checkpoints/       saved neural networks (for SAC training runs)
└── code/              a copy of the code that produced this run
```

### `summary.json`: the quick answer

The most useful fields:

| Field | Meaning |
|---|---|
| `status` | `completed`, `interrupted` (you stopped it) or `crashed` |
| `episodes`, `finishes`, `finish_rate` | how many attempts, and how many finished |
| `best_time_ms`, `median_time_ms` | fastest and typical finish time |
| `track_pct.mean` | average % of the track covered |
| `track_pct.ci95` | how uncertain that average is (see [page 7](07-fair-experiments.md)) |
| `done_reasons` | how many episodes ended each way |
| `last_window` | the same numbers for only the last 100 episodes, which is what matters for a run that was learning |
| `distinct_outcomes` | how many different results there were. A deterministic driver should have **1**. |
| `step_gaps` | should be **0**. Anything else means the run wasn't clean (see [page 9](09-troubleshooting.md)). |

### `episodes.csv`: every attempt

Open it in a spreadsheet, or with pandas (from the `TMRC` folder):

```python
import pandas as pd
df = pd.read_csv("results/<run>/episodes.csv")
df["track_pct"].plot()                   # progress over the run
df[df.finished]["race_time_ms"].min()    # best finish time
```

(pandas isn't included; install it with `uv add pandas`, or use a spreadsheet.)

Columns include `done_reason`, `race_time_ms`, `distance_m`, `track_pct`,
`score` (total reward), `mean_speed_kmh`, `max_speed_kmh`, `agent_ms_max` (your
slowest decision) and `replay` (the replay file for that episode, if one was saved).

When several game instances ran from separate processes, each has its own
`episodes-instance-N.csv`.

### `manifest.json`: how to repeat it

Records the seed, every setting (`TM_*` variables and the run's config), a
fingerprint (hash) of every code file, the map and track data, which saved
network was loaded, and the hardware. `code/` has the actual files. To repeat a
run, use the same code and the same settings, including `TM_SEED`.

### `metrics/`

- `perf.csv`: speed of training (steps per second, and how many times faster than real time)
- `learner.csv`: SAC learning statistics (see [page 5](05-ways-to-learn.md#the-included-trainer-sac))
- anything your own code logs with `run.log_metrics(...)`

## Comparing runs

From the `TMRC` folder on your computer (no game needed):

```bash
uv run agent/compare_runs.py                         # all runs, best first
uv run agent/compare_runs.py --kind eval             # only evaluations
uv run agent/compare_runs.py --map A03-Race
uv run agent/compare_runs.py --show <run>            # one run's details
uv run agent/compare_runs.py --diff <run A> <run B>  # what was different between two runs
```

You can name runs by any unique start or end of their folder name.

`--diff` is the tool for "why did these two runs give different results?". It
lists every setting, code file and game file that differs.

## Replays

A replay records **every step** of an episode: the keys pressed, when, and where
the car went. Which episodes get one depends on `TM_SAVE_REPLAYS`:

| Value | Saves |
|---|---|
| `best` (default) | every episode that beat the best so far in this run, so you can see how it improved |
| `finished` | the above, plus every finish |
| `all` | every episode (uses a lot of disk space on long runs) |
| `none` | nothing |

Each replay is two files:
- `*.npz`: the data, for Python
- `*.inputs.txt`: the same keys as a TMInterface input script

### Looking inside a replay

Start Python from the `TMRC/agent` folder (`cd TMRC/agent && uv run python`), so
the framework can be imported:

```python
from experiments import Replay
r = Replay.load("../results/<run>/replays/<file>.npz")
print(r.meta)          # which run, episode, result, settings
r.position             # where the car was after each step, shape (steps, 3)
r.inputs               # keys pressed each step: columns left, right, accelerate, brake
r.obs                  # what the driver saw each step
r.race_time_ms         # game time after each step
```

For example, to draw the route the car took (needs `matplotlib`):

```python
import matplotlib.pyplot as plt
plt.plot(r.position[:, 0], r.position[:, 2]); plt.axis("equal"); plt.show()
```

### Checking a replay reproduces

Because the game is deterministic, sending the replay's keys again should
produce the same route. To check:

```bash
TM_MODE=verify TM_REPLAY=<run>/replays/<file>.npz docker compose up --build
```

It prints `MATCH` or `DIVERGED`, and saves the details next to the replay as
`<file>.verify.json`. A `DIVERGED` result means something in the setup has
changed or isn't reliable. [Page 9](09-troubleshooting.md) has suggestions.

### Watching a replay in the game

The `.inputs.txt` file is in TMInterface's input-script format, so you can load
it into the game by hand through TMInterface. The exact timing hasn't been
checked in-game yet: if the car drifts compared with the recording, the inputs
may need shifting by one game tick (10ms). `Replay.to_tmi_script(offset_ms=10)`
writes a shifted version.

## Where to go next

- [7. Running fair experiments](07-fair-experiments.md)
- Back to the [index](README.md)
