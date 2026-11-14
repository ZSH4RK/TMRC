# 11. Entering the competition

The Trinity-RL competition calls the **same `decide(game_state, wall_distances)`
function** as this framework, with the same rays and the same decision every
50 ms of game time. A driver you build and test here is submitted as it is.

## Joining

Follow the steps in [competition-admin](https://github.com/Trinity-RL/competition-admin).
You'll get access to [`tmrc-demo`](https://github.com/Trinity-RL/tmrc-demo), a
complete working entry, and your own private `submission-<username>` repository.

## Submitting a driver from this framework

1. Copy your agent file into your submission repo as `agent/agent.py`. It only
   needs `decide()`: the competition imports it and calls that.
2. Copy anything it imports, such as helper modules or model weights, into
   `agent/` as well. If it uses `track_features`, copy `track_features.py` too:
   it finds the competition's track data and loads the map being raced (see below).
3. If it needs extra Python packages (for example `torch`), add them to the
   `uv pip install` line in the submission's `Dockerfile`. The competition runs
   Python 3.12, like this framework.
4. Test it the way the competition runs it (below), then push to `main`. That
   queues an evaluation, and the result appears on the leaderboard.

## The rules

- **Maps:** A01-Race, A02-Race and A03-Race, in that order, **one attempt each**.
  All three have track data, so the rays work on every map. Train on all three:
  set `TM_MAP_NAME` (e.g. `TM_MAP_NAME=A01-Race`), see [page 8](08-settings-reference.md).
- **Time limit:** 5 minutes of **game time** per map. It's measured on the race
  clock, so a slow `decide()` doesn't cost you race time (it does make the
  evaluation take longer).
- **No "stuck" rule:** the attempt runs until you finish or time runs out. Here,
  episodes end after 3 s without progress; there, a car that's stuck stays stuck.
- **A fresh start on each map:** `agent.py` is imported again for each map, with
  the environment variable `TM_MAP_NAME` set to it (e.g. `A02-Race`). Variables at
  the top of your file start clean, and `track_features` loads that map's centre line.
- **Errors:** if `decide()` raises an error or doesn't return a dict, the car
  accelerates for that step and the error is counted in your results. Here, an
  error stops the run so you'll see it, so test with `TM_MODE=eval` first.
- **Ranking:** per map, finishes by time, then non-finishes by how far along
  the track they got: the same race ranking this framework uses.

## Checking before you submit

In this framework, on each map:

```bash
TM_MODE=eval TM_EVAL_AGENT=agent:my_driver.py TM_EVAL_EPISODES=3 TM_MAP_NAME=A01-Race docker compose up --build
```

`summary.json` shows whether it finishes and how fast.

Then in your submission repo, run the competition's own harness on it:

```bash
docker build -t my-driver .
mkdir -p results && docker run --rm -v "$PWD/results:/results" my-driver
```

It writes the same results file an evaluation produces, including the number of
`decide()` errors and the keys your driver pressed on each map.
