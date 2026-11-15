# 5. Ways to learn

This page gives an overview of approaches. None of them is "the" way. Each
trades off simplicity, speed and how good it can get. Read the ones that interest you.

| Approach | In one sentence | Included here |
|---|---|---|
| [Hand-written rules](#hand-written-rules) | You decide what the car does in each situation | `examples/wall_follower_agent.py` |
| [Trial-and-error search](#trial-and-error-search) | Randomly tweak numbers and keep what works | `examples/hill_climb.py` |
| [Reinforcement learning](#reinforcement-learning) | The car learns from rewards over many episodes | `train_sac_vec.py` (SAC) |
| [Learning from demonstrations](#learning-from-demonstrations) | Copy how someone else drove | not included, but replays have the data |
| [Something else](#something-else) | Path planning, mixing methods, … | — |

---

## Hand-written rules

You write the logic yourself: "if the left wall is close, steer right".

- **Good:** you understand exactly why the car does what it does. It's quick to
  try and needs no training time.
- **Hard:** fast driving needs many rules that interact, and tuning them by hand is slow.
- **Start with:** [page 3](03-writing-an-agent.md) and `examples/wall_follower_agent.py`.

## Trial-and-error search

Give your driver some adjustable numbers (**parameters**), then let the computer
search for good values: try a change, drive an episode, keep it if the result
was better. This family is often called **black-box optimisation**. Hill
climbing, random search, evolutionary algorithms and CMA-ES all belong to it.

- **Good:** simple to understand, and works with *any* driver that has numbers
  in it, including your hand-written rules. TrackMania is deterministic, so a
  single episode is a fair test of one set of numbers.
- **Hard:** it gets slow when there are many parameters (thousands or more).
- **Start with:** `examples/hill_climb.py`. It tunes 30 numbers of a
  "linear" driver. You could point the same loop at your own rule-based driver's
  numbers instead.

## Reinforcement learning

**Reinforcement learning (RL)** learns by doing. The driver (the **policy**)
acts, gets a **reward** each step, and gradually changes so that it collects
more reward over an episode. Nobody tells it the right action; it works out
which actions lead to more reward later on.

Some words you'll meet:
- **Policy:** the thing that picks actions (here, a neural network).
- **Exploration:** trying actions the policy isn't sure about, to discover better ones.
  This is why training scores look worse and noisier than evaluation scores.
- **Replay buffer:** a memory of recent steps that the learner studies.
- **Value / Q function:** the network's guess of how much reward is still to come.

### The included trainer: SAC

`agent/train_sac_vec.py` uses **Soft Actor-Critic (SAC)**, a popular RL
algorithm for continuous actions. It runs several game instances at once and
learns from all of them together.

```bash
TM_MODE=train docker compose up --build          # train, from scratch
TM_LOAD_CHECKPOINT=latest TM_MODE=train docker compose up --build   # carry on from where it stopped
```

- Saved networks go in `agent/tmp/sac/` (best so far) and `agent/tmp/sac_latest/`
  (most recent), and are also copied into each run's `checkpoints/` folder.
- `[PB]` lines in the output announce new personal bests (fastest finish, furthest
  distance) within the run. They are kept in the run's own `pb.json`
  (`results/<run>/pb.json`); each run starts afresh.
- To test the trained network properly (no exploration noise):
  `TM_MODE=eval TM_EVAL_AGENT=sac:tmp/sac docker compose up --build`

How to tell whether learning is working, from the run's `metrics/learner.csv`:

| Column | What to look for |
|---|---|
| `entropy` | how random the policy is. With `TM_AUTO_ENTROPY` on it is steered towards `TM_TARGET_ENTROPY` (-2). |
| `temperature` | how strongly randomness is rewarded. Entropy tuning raises it when the policy is less random than the target and lowers it when more. |
| `lr` | the current learning rate; falls over time if `TM_LR_DECAY_UPDATES` is set. |
| `q_mean` | the network's estimate of future reward. Should rise as driving improves; values that explode suggest instability. |
| `critic_loss`, `value_loss` | how wrong its estimates are. Noisy; look at the trend, not single values. |
| `actor_loss` | not very meaningful on its own |

And from `episodes.csv`, the thing that really matters: `track_pct` and
`race_time_ms` going the right way over time.

If scores peak and then slowly get worse, the best network is still safe: it's
saved by its 100-episode average in `agent/tmp/sac/` and the run's
`checkpoints/best/`. To stop the decline, try `TM_LR_DECAY_UPDATES` (see
[page 8](08-settings-reference.md#sac-training)).

RL often needs **hours** and many episodes before it gets good, and results vary
between runs with different seeds (see [page 7](07-fair-experiments.md)).

### Writing your own RL algorithm

`TMNFEnv` ([page 4](04-the-environment.md)) has the same shape as standard RL
environments, so you can implement PPO, DQN or anything else yourself, following
the structure of `examples/hill_climb.py`. Libraries that expect a Gymnasium
environment would need a small adapter class. None is included yet.

## Learning from demonstrations

Also called **imitation learning** or **behaviour cloning**: record good driving,
then train a model to predict "in this situation, press these keys".

Every saved replay (`results/<run>/replays/*.npz`) contains, step by step, the
observation (`obs`) and the keys pressed (`inputs`), which is exactly the data
this needs. You could learn from your best rule-based driver's replays, or use
them to give an RL agent a head start. See [page 6](06-results-and-replays.md)
for loading replays.

## Something else

The framework doesn't assume any particular approach. Some other directions people take:

- **Planning a racing line** from the track data in `Tracks/`, then steering to follow it.
- **Combining methods:** rules for safety plus learning for speed, or search to
  tune a learned driver's settings.
- **Changing the observation or reward** to make learning easier.

Whatever you try, recording it with a `Run` ([page 4](04-the-environment.md#recording-what-your-learner-does))
means you can compare it fairly with everything else.

## Where to go next

- Compare approaches fairly: [7. Running fair experiments](07-fair-experiments.md)
- Back to the [index](README.md)
