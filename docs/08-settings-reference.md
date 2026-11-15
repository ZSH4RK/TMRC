# 8. Settings reference

Settings are **environment variables**. Put them before the command:

```bash
TM_MODE=eval TM_EVAL_EPISODES=10 docker compose up --build
```

Defaults are set in `TMNF-Docker/docker-compose.yml`. You can also edit that file.
Every setting starting with `TM_` is saved in each run's `manifest.json`.

## What to run

| Setting | Default | Meaning |
|---|---|---|
| `TM_MODE` | `train` | `train`: SAC with several instances · `train_single`: SAC with one instance · `eval`: test a driver · `verify`: check a replay reproduces · `script`: run your own file · `drive` (or any other value): drive forever with `TM_AGENT` |
| `TM_EVAL_AGENT` | `sac:tmp/sac` | eval mode: which driver (see *Driver specs* below) |
| `TM_EVAL_EPISODES` | `5` | eval mode: how many episodes in total |
| `TM_AGENT` | `agent:my_agent.py` | drive mode: which driver |
| `TM_REPLAY` | — | verify mode: the replay file, e.g. `<run>/replays/<file>.npz` |
| `TM_VERIFY_TOL_M` | `0.01` | verify mode: how far (metres) the car may drift before it counts as diverged |
| `TM_SCRIPT` | — | script mode: your Python file, e.g. `agent/examples/hill_climb.py` |

### Driver specs

Used by `TM_EVAL_AGENT` and `TM_AGENT`:

| Spec | Driver |
|---|---|
| `agent:<file.py>` | any file with a `decide()` function, relative to `agent/` |
| `sac:<folder>` | a saved SAC network, driven without exploration noise, e.g. `sac:tmp/sac` |
| `forward` | always accelerate, never steer |
| `random` | random keys (uses the seed) |

## Recording

| Setting | Default | Meaning |
|---|---|---|
| `TM_RUN_NAME` | — | a label added to the run's folder name |
| `TM_SEED` | random | fixes all random numbers; the one used is always saved |
| `TM_SAVE_REPLAYS` | `best` | `best` · `finished` · `all` · `none` (see [page 6](06-results-and-replays.md#replays)) |
| `TMRC_GIT_COMMIT` | `unknown` | set at build time to record the git commit: `TMRC_GIT_COMMIT=$(git -C TMRC rev-parse HEAD) docker compose up --build` |

## The game

| Setting | Default | Meaning |
|---|---|---|
| `TM_NUM_INSTANCES` | `5` | how many copies of the game to run at once |
| `TM_GAME_SPEED` | `5.0` | how many times faster than real time |
| `TM_STEP_MS` | `50` | game time between decisions (a multiple of 10). A driver trained at one value drives differently at another. |
| `TM_MAP_NAME` | `A03-Race` | which map; needs a matching folder in `Tracks/` |
| `TM_BASE_PORT` | `8775` | network port of instance 0 (instance *i* uses 8775 + *i*) |

## Watching

| Setting | Default | Meaning |
|---|---|---|
| `NOVNC_ENABLE` | `true` | serve the game view in a browser |
| `TM_VIEW_INSTANCES` | `1` | how many instances can be watched. Instance *i* is at port 6080 + *i*: `http://localhost:6080`, `http://localhost:6081`, … |
| `VNC_BIND` | `127.0.0.1` | set to `0.0.0.0` to watch from another computer on your network |

## SAC training

| Setting | Default | Meaning |
|---|---|---|
| `TM_LOAD_CHECKPOINT` | `false` | `true`: continue from the best saved network · `latest`: from the most recent one |
| `TM_N_STEPS` | `0` | stop after this many steps, summed over all instances; `0` trains until stopped (Ctrl-C or `docker stop`) |
| `TM_UPDATES_PER_STEP` | `1.0` | learning updates per step of experience |
| `TM_EVAL_ONLY` | `false` | drive the loaded network inside the trainer without learning (`eval` mode is the cleaner way to test) |
| `TM_TORCH_THREADS` | `2` | CPU threads for the learner |
| `TM_WALL_PENALTY` | `1.0` | reward taken off for each step that ends touching a wall; `0` turns it off. Changing it changes every score, so the next training run copies the saved best network into its results folder (`checkpoints/previous-best/`) and starts its best score afresh |
| `TM_REPLAY_SIZE` | `1000000` | steps of experience the learner keeps (about 1,500 A03 episodes with 6 instances). Too small and it forgets mistakes it stopped making, then makes them again. Uses about 125 MB per million |
| `TM_LR` | `0.0003` | learning rate of all the networks |
| `TM_LR_DECAY_UPDATES` | `0` | learning updates over which the learning rate falls in a straight line to `TM_LR` × `TM_LR_FINAL`, then stays there. `0` keeps it constant. Training past its best and getting worse is the sign to try it: e.g. set it to the `updates` count in `learner.csv` where scores peaked |
| `TM_LR_FINAL` | `0.1` | where the decay ends, as a fraction of `TM_LR` |
| `TM_AUTO_ENTROPY` | `true` | adjust how much the policy explores so its `entropy` tracks `TM_TARGET_ENTROPY`. `false` uses the original fixed amount |
| `TM_TARGET_ENTROPY` | `-2` | the entropy aimed for; lower = less random. The usual choice is minus the number of actions |

## Constants in the code

Some things are constants in code, not settings. Change them in the file and rebuild:

| What | Where |
|---|---|
| Reward amounts, episode time limit, "stuck" rule | top of `agent/env.py` |
| Ray angles, observation scaling, key deadzones | `agent/track_features.py` |
| SAC network size, batch size, discount, reward scale | `agent/sac/agent.py` |

All of these are saved in each run's `manifest.json` under `config.env`, so
runs with different values can be told apart.
