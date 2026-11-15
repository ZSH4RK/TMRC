# 3. Writing an agent

An **agent** is a Python file with one function, `decide`. The framework calls
it every step (every 50ms of game time) and presses whatever keys it returns.

```python
def decide(game_state, wall_distances):
    return {"left": False, "right": False, "accelerate": True, "brake": False}
```

That's a complete agent (it's `agent/examples/forward_agent.py`). Everything
else is up to you: rules, maths, a neural network, a lookup table, anything.

## What your function is given

### `game_state`: where the car is and how it's moving

```python
{
    "position":  (x, y, z),     # where the car is, in metres. y is height.
    "velocity":  (vx, vy, vz),  # how fast it's moving in each direction, metres/second
    "direction": (dx, dy, dz),  # which way the car is pointing (length 1)
}
```

Speed in km/h, for example:

```python
import math
vx, vy, vz = game_state["velocity"]
speed_kmh = math.sqrt(vx*vx + vy*vy + vz*vz) * 3.6
```

### `wall_distances`: what the car can "see"

The framework sends out 7 invisible **rays** from the car, like a fan of
measuring tapes, and reports how far each one goes before hitting a wall:

```
            -10°  0°  +10°
        -45°   \  |  /   +45°
               \ | /
      -90° ─────[car]───── +90°
```

You get a list of 7 pairs, in the order of the angles above
(`[-90, -45, -10, 0, 10, 45, 90]`):

```python
[(11.1, "left"), (15.7, "left"), (64.0, "left"), (None, None), (64.1, "right"), (15.7, "right"), (11.1, "right")]
```

- The **first** number is the distance to the wall in metres, or `None` if the ray
  didn't hit anything within 500m.
- The **second** says which wall it hit, `"left"` or `"right"`.
- The **middle** ray (angle 0) points straight ahead.

> **Tip:** use the `"left"`/`"right"` labels rather than assuming which angle
> points which way. The code comments in different files disagree about the
> sign of the angles, and the labels always tell you which wall it is.

## What your function returns

A dictionary with four True/False keys:

| Key | Meaning |
|---|---|
| `"left"` | steer left |
| `"right"` | steer right |
| `"accelerate"` | throttle on |
| `"brake"` | brake (or reverse, when stopped) |

TrackMania Nations only has keyboard controls here, so there's no "half left".
You can press any combination, but `left` and `right` together cancel out. Any
key you leave out counts as not pressed.

## Try your agent

Put your file anywhere inside `TMRC/agent/` (for example `TMRC/agent/my_driver.py`),
then from `TMNF-Docker/`:

```bash
TM_MODE=eval TM_EVAL_AGENT=agent:my_driver.py TM_EVAL_EPISODES=3 TM_NUM_INSTANCES=1 docker compose up --build
```

The path after `agent:` is relative to the `agent/` folder. Remember `--build`:
without it your changes aren't in the container.

Other ways to run it:
- **Drive forever** (good for watching): `TM_MODE=drive TM_AGENT=agent:my_driver.py docker compose up --build`
- **Several cars at once:** add `TM_NUM_INSTANCES=3`

## Ideas to start from

These are just suggestions. Ignore them if you have your own plan.

- **Read `agent/examples/wall_follower_agent.py`.** It steers away from whichever
  wall is closer and brakes before walls. Change its numbers and compare results.
- **Print what your agent sees.** Add `print(wall_distances)` inside `decide`, run
  it, and watch the output while looking at the car in the browser.
- **Use the observation the trainers use.** `track_features.build_state(game_state, wall_distances)`
  turns the inputs into 15 numbers scaled to about -1..1 (see
  [page 4](04-the-environment.md#the-observation)). `examples/linear_agent.py` does this.
- **Use the track's centre line.** `track_features.progress_along_track(x, z)` says how
  far along the track the car is, and `track_features.nearest_midpoint(x, z)`
  gives the nearest point on the centre line plus one a bit further ahead. You could aim at it.
- **Remember things between steps.** Variables at the top of your file keep their
  values between calls, so you can measure things like "how long have I been steering left?".
  Each game instance gets its own copy of your file, so instances don't interfere.

## Things to know

- **Your function should be quick.** The game waits for it, so a slow `decide`
  makes tests take longer (but doesn't change the result). How long it took is
  recorded in `agent_ms_mean` / `agent_ms_max` (see [page 6](06-results-and-replays.md)).
- **Errors stop the run.** If `decide` crashes, the error is printed and the run
  is marked `crashed`.
- **Wall data exists for A01-Race, A02-Race and A03-Race**, the competition maps.
  Pick one with `TM_MAP_NAME` (default `A03-Race`). Other maps need their own
  folder in `Tracks/`: its README explains how to make one.

## Where to go next

- How well did it do? [6. Results and replays](06-results-and-replays.md)
- Have the computer improve it for you: [5. Ways to learn](05-ways-to-learn.md)
- Back to the [index](README.md)
