# 4. The environment

If you're writing code that **learns**, with any method, you'll want the
**environment**: `TMNFEnv` in `agent/env.py`. It turns the game into the
`reset()` / `step()` loop that most learning code expects, the same shape as
OpenAI Gym / Gymnasium environments.

`decide()` agents from [page 3](03-writing-an-agent.md) don't need this page.
The framework runs the loop for them.

## The loop

```python
obs = env.reset()                   # put the car on the start line; returns the first observation
done = False
while not done:
    action = ...                    # your method picks an action from obs
    obs, reward, done, info = env.step(action)
```

| Call | What it does |
|---|---|
| `env.reset()` | Restarts the race and returns the first **observation** |
| `env.step(action)` | Steps with a continuous `action` (see below) |
| `env.step_inputs({"left": ..., "right": ..., "accelerate": ..., "brake": ...})` | Steps with keys directly, if your method thinks in keys |
| `env.close()` | Disconnects from the game |

`step` returns four things:
- `obs`: the new observation
- `reward`: a number saying how good that step was (see below)
- `done`: `True` when the episode has ended
- `info`: a dictionary with extra details (see the end of this page)

After each `reset()` or `step()`, `env.last_game_state` and `env.last_wall_distances`
hold the raw inputs a `decide()` agent would get, so you can use those instead of `obs`.

## The observation

`obs` is a NumPy array of **14 numbers**, built by `track_features.build_state()`.
Each is scaled to roughly -1..1, which suits neural networks:

| Index | Meaning | Scaling |
|---|---|---|
| 0–6 | wall distance along each of the 7 rays | metres ÷ 100, capped at 1 (no hit = 1) |
| 7, 8 | velocity x, z | m/s ÷ 100 |
| 9, 10 | facing direction x, z | already -1..1 |
| 11 | angle between where the car points and the track direction | degrees ÷ 180 |
| 12 | angle between where the car is *moving* and the track direction | degrees ÷ 180 |
| 13 | distance from the centre line | metres ÷ 20 |

You don't have to use it. You could build your own observation from
`env.last_game_state` and `env.last_wall_distances`.

## The action

`env.step(action)` takes two numbers, each between -1 and 1:

| | -1 | 0 | +1 |
|---|---|---|---|
| `action[0]`: steer | full left | straight | full right |
| `action[1]`: throttle | brake | coast | accelerate |

The game only has keys, so these are converted (`track_features.action_to_inputs`):
anything beyond ±0.1 counts as pressing that key. If your method picks from a
handful of choices instead (like "left + accelerate"), use `env.step_inputs()` directly.

## The reward

The reward is what reinforcement learning maximises. In this environment, each step gives:

- **+1 per metre** moved forward along the track's centre line (negative if going backwards)
- **-0.0012 per millisecond** of game time, so -0.06 for each 50ms step. Faster is better.
- **+100** for crossing the finish line

The constants are at the top of `agent/env.py`. Changing them changes what
"good" means, and you're welcome to experiment. Rewards from runs with
different reward settings can't be compared with each other, though. That's why
the framework ranks results by **race result** (finished? how fast? how far?)
instead (see [page 7](07-fair-experiments.md)).

## When an episode ends

`done` becomes `True`, and `info["done_reason"]` says why:

| `done_reason` | Meaning |
|---|---|
| `finished` | crossed the finish line |
| `timeout` | 120 seconds of game time |
| `stuck` | less than 1 metre of progress in 3 seconds |

## Recording what your learner does

Wrap the environment in `RecordingEnv` and every episode is recorded, the same
as for the included trainers:

```python
from env import TMNFEnv
from experiments import RecordingEnv, Run

run = Run.create("train", {"policy": "my-method", "learning_rate": 0.01})   # put your settings here
env = RecordingEnv(TMNFEnv(port=port), run, policy="my-method")
...
run.log_metrics("my_method", {"iteration": i, "loss": loss})   # any numbers you want over time
run.finalize("completed")
```

Anything you put in `Run.create(..., config)` is saved in the run's `manifest.json`,
so you'll always know what settings produced a result.

## A complete example

`agent/examples/hill_climb.py` is a full, short learning script: it creates a
run, starts the game, wraps the environment, and improves a simple policy by
trial and error. Copy it as a starting point for any method, or don't.

Run your own script with:

```bash
TM_MODE=script TM_SCRIPT=agent/examples/hill_climb.py docker compose up --build
```

## Everything in `info`

| Key | Meaning |
|---|---|
| `done_reason` | why the episode ended (only when `done`) |
| `race_time_ms` | game time, in milliseconds |
| `distance_m` | metres along the centre line this episode |
| `track_pct` | the same as a percentage of the track |
| `position`, `velocity`, `direction` | the car's state after this step |
| `inputs` | the keys that were pressed this step |
| `input_time_ms` | the game time when those keys were sent |
| `step_gap` | `True` if game time jumped by more than one step. This shouldn't happen; see [page 9](09-troubleshooting.md) |
| `agent_ms` | how long your code took to decide, in real milliseconds |
| `game_ms` | how long the game took to run the step, in real milliseconds |
| `episode_row` | (with `RecordingEnv`, when `done`) the row recorded for this episode |

## Where to go next

- Choose a learning approach: [5. Ways to learn](05-ways-to-learn.md)
- Back to the [index](README.md)
