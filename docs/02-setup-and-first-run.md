# 2. Setup and your first run

## What you need

- **Linux** with an **NVIDIA graphics card** (the game is drawn on the GPU).
- **Docker** with the **NVIDIA Container Toolkit**, so containers can use the GPU.
  Check it works with:
  ```bash
  docker run --rm --gpus all ubuntu nvidia-smi
  ```
  If that prints a table describing your GPU, you're ready.
- **git**, and **[uv](https://docs.astral.sh/uv/)** (a Python tool) if you want to
  look at results or run tests on your own computer rather than in the container.
- The base TrackMania image, `tmnf-test:latest`. Building it from scratch is
  described in the main `TMNF-Docker` README. It's slow and needs an NVIDIA
  registry login, so if someone can give you the image, use that.

## Get the code

The code is in three repositories. They must sit **next to each other** in one folder:

```
code/
├── TMNF-Docker/      the Docker setup (TMRC is inside it)
│   └── TMRC/         the Python framework, agents and these docs
└── Tracks/           wall and centre-line data for each map
```

```bash
git clone --recurse-submodules https://github.com/ZSH4RK/TMNF-Docker.git
git clone https://github.com/Trinity-RL/Tracks.git
```

`Tracks/` is needed because the car "sees" walls by measuring against a map of
the track, and that map lives there.

## Your first run: watch a simple driver

From the `TMNF-Docker` folder:

```bash
TM_MODE=eval TM_EVAL_AGENT=forward TM_EVAL_EPISODES=3 TM_NUM_INSTANCES=1 docker compose up --build
```

What that means:
- `TM_MODE=eval`: test a driver (rather than train one)
- `TM_EVAL_AGENT=forward`: the driver to test; `forward` always accelerates and never steers
- `TM_EVAL_EPISODES=3`: three attempts
- `TM_NUM_INSTANCES=1`: one copy of the game
- `--build`: rebuild the image first, which picks up any code changes

The first build takes a while. Then you'll see the game start (it waits about
25 seconds for the game to load) and lines like:

```
[eval 0] episode 1: stuck time=6.35s dist=212m (17.4%)
```

### Watch it

Open **http://localhost:6080** in a browser. The password is `mypasswd`. You'll
see the game, with the car driving into the first wall.

### See what was recorded

When it finishes, there's a new folder in `TMNF-Docker/TMRC/results/`, named
something like `20261112-153000-eval`. Open `summary.json` in it. All three
episodes should be identical, because the game is deterministic and so is `forward`.

From the `TMRC` folder you can also print a table of all your runs:

```bash
cd TMRC
uv run agent/compare_runs.py
```

## Stopping

Press **Ctrl+C** in the terminal, or run `docker compose down` from another
terminal. The framework gets up to 60 seconds to save its work.

## Important: code changes need `--build`

Your Python code is **copied into** the image when it's built. If you edit a file
and run `docker compose up` without `--build`, the container runs the **old** code.

Two folders are shared with your computer instead of copied, so changes there
show up straight away:
- `TMRC/results/`: everything recorded
- `TMRC/agent/tmp/`: the SAC trainer's saved neural networks

## Where to go next

- Write your own driver: [3. Writing an agent](03-writing-an-agent.md)
- Train one: [4. The environment](04-the-environment.md) and [5. Ways to learn](05-ways-to-learn.md)
- Back to the [index](README.md)
