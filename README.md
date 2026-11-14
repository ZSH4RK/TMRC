<img width="512" height="512" alt="TMRClogo" src="https://github.com/user-attachments/assets/6821a4b4-25c0-4267-ad84-d8776ba37e0e" />

# TMRC
The Trinity Machine Racing Championship is a competition run at Trinity School, to see who can create the fastest AI bot in Trackmania Nations Forever. The competitors will have access to the TMInterface game state (eg. Acceleration, Velocity, Floor contact per wheel, Wall contact flag), the .gbx map file and a compressed image, that can be used for path tracing. The model will be submitted as a python agent.py file to be run by a central server that runs all models in equal conditions for the overall race result.


## Plan 
1-3 Proof of Concept <br>
MVP 4-5 <br>
Nice To Have 6- <br>

- [X] TMNF running with TMInterface loading onto a specific map (A01) from Commandline
- [X] Get a basic model that just drives forward running from Command Line
- [X] Use Docker to package the model and allow participants to use their own models (`TM_MODE=eval TM_EVAL_AGENT=agent:<file>`, and [submission repos](docs/11-competition.md))
- [X] Logging mechanism for results eg. Distance Covered or Racetime, Timestamp, inputs performed (`results/<run>/`: episodes, replays + TMInterface input scripts; see below)
- [X] Allow results to be sent back to host Computer and saved (bind-mounted `results/`)
- [X] Live Leaderboard on each track (competition-admin's evaluation workflow rebuilds it on every submission)
- [X] Parallelize running TMNF while allowing the model to learn from all instances at once (`TM_NUM_INSTANCES`, `train_sac_vec.py`)
- [X] Track data (walls + centre line) for all three competition maps: A01, A02, A03 (`agent/trace_walls.py`)

### Results, replays and testing
**New here? Start with [docs/README.md](docs/README.md)**: step-by-step guides for writing agents, training, and reading results.

Every way of running the game records a **run** under `results/<run_id>/` (bind-mounted to `TMRC/results/` on the host), so any result can be compared with others and reproduced:

| File | Contents |
|---|---|
| `manifest.json` | what ran: code hash (+ git commit), config, every `TM_*` setting, seed, map/track/plugin hashes, loaded checkpoint hashes, hardware |
| `code/` | a copy of the agent code, track code and game plugins/config that produced the run |
| `episodes*.csv` | one row per episode: result, time, distance, speed, step gaps, agent latency, replay file |
| `replays/*.npz` + `*.inputs.txt` | recorded episodes (inputs + trajectory) and a TMInterface input script for each |
| `metrics/` | `perf.csv` (throughput) and `learner.csv` (SAC losses, entropy, Q) for training runs |
| `checkpoints/` | the weights the run ended with |
| `summary.json` | finish rate, best/median time, mean ± 95% CI distance, done reasons, health counters |

Pick what runs with `TM_MODE` (see `docker-compose.yml`):
```bash
TM_MODE=train docker compose up                                         # SAC training (default)
TM_MODE=eval TM_EVAL_AGENT=sac:tmp/sac TM_EVAL_EPISODES=5 docker compose up
TM_MODE=eval TM_EVAL_AGENT=agent:my_agent.py docker compose up           # any decide() file
TM_MODE=verify TM_REPLAY=<run_id>/replays/<file>.npz docker compose up   # does the replay reproduce?
TM_MODE=drive TM_AGENT=agent:my_agent.py docker compose up               # drive forever, recorded
```
`TM_RUN_NAME` labels a run, `TM_SEED` fixes its seed (random and recorded otherwise), `TM_SAVE_REPLAYS` = `best` (default: each improvement) | `finished` | `all` | `none`.

Compare results on the host:
```bash
uv run agent/compare_runs.py                    # leaderboard of every run, race order
uv run agent/compare_runs.py --kind eval --map A03-Race
uv run agent/compare_runs.py --diff <run A> <run B>   # what changed: seed, config, env, code files, game files
uv run pytest                                   # framework tests (no game needed)
```

### Technologies
* Python
* TMInterface
* Docker
* Commandline


### Steps
#### 1 - Command Line Loading
I used TMModloader to inject TMInterface into Trackmania Nations Forever. I found a Plugin created by the Linesight team, written in AngelScript (The only language that TMModloader and TMInterface support) that connects to a reverse engineered TMInterface API, since the real one was depreciated when TMModloader was released. Using this API I can gather the game state and send inputs to Trackmania from running `agent.py` in the commandline. <br>

#### 2 - Simplest Possible Model
My model recieves the game state from the `Python_Link.as` plugin and just drives forward. It also automatically sends the commands `set autologin 1` (first profile), `set custom_port <port>` and `map A01-Race.Challenge.Gbx` in the TMInterface commandline. This skips the offline only and profile selection screenm and loads directly into the selected map.
