# 11. Entering the competition

The Trinity-RL competition runs the **same `decide(game_state, wall_distances)`
interface** as this framework. A driver you build and test here can be submitted
as it is.

## Joining

Follow the steps in [competition-admin](https://github.com/Trinity-RL/competition-admin).
You'll get access to `tmrc-demo` (a working example) and your own private
`submission-<username>` repository.

## Submitting a driver from this framework

1. Copy your agent file into your submission repo as `agent/agent.py`. Keep the
   runner code below `decide()` in the template's `agent.py`: paste your
   `decide()` and its helpers above it.
2. Copy anything your file imports, such as helper modules or model weights,
   into `agent/` as well.
3. If it needs extra Python packages (for example `torch`), add them to the
   `pip install` line in the submission's `Dockerfile`.
4. Push to `main`. That queues an evaluation, and the result appears on the leaderboard.

## What's the same, and what's different

**The same as in this framework:**
- the `game_state` keys and `wall_distances`: the same rays, angles and track data
- one decision every 50 ms of game time
- the keys you return

**Different:**
- **Maps:** A01, A02 and A03-Race. Track data exists for **A03-Race only**, so on
  the others every ray is `(None, None)`. A driver that relies on rays needs a fallback.
- **One attempt per map**, with a 5-minute limit. There's no "stuck" rule ending
  the attempt early.
- **No `track_features`:** the competition only gives you `game_state` and
  `wall_distances`. If your driver uses `track_features` (like `my_agent.py` and
  `linear_agent.py`), copy `track_features.py` in too. Note that it loads one
  map's centre line when imported (`TM_MAP_NAME`, default A03-Race).
- **Errors:** if `decide()` raises an error, the car accelerates for that step
  and the error is counted in your results. Test with `TM_MODE=eval` first:
  here, an error stops the run so you'll see it.

## Checking before you submit

```bash
TM_MODE=eval TM_EVAL_AGENT=agent:my_driver.py TM_EVAL_EPISODES=3 docker compose up --build
```

`summary.json` shows whether it finishes and how fast. The competition's
ranking is the same race ranking this framework uses: finishes by time, then
non-finishes by how far they got.
