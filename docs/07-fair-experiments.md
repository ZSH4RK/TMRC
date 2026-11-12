# 7. Running fair experiments

It's easy to fool yourself: you change something, the next run looks better, and
you conclude your change helped. Sometimes it did; sometimes you got lucky.
This page covers habits that help you tell the difference. They apply whatever
approach you use.

## Change one thing at a time

If you change the reward *and* the learning rate *and* the observation, and the
result improves, you won't know which change helped. Where you can, change one
thing per run.

`compare_runs.py --diff <A> <B>` lists everything that differs between two runs.
It's worth checking before you draw conclusions.

## Label your runs

```bash
TM_RUN_NAME=bigger-brake-margin TM_MODE=eval ... docker compose up --build
```

The label goes into the folder name, so you can find it in six weeks' time.

## Judge by evaluation, not by training scores

While a learning method trains, it **explores**: it deliberately tries
actions it isn't sure about. So training episodes look worse and vary more than
the driver really is. And the driver keeps changing, so each training episode is
from a slightly different driver.

To measure a driver, **freeze it and evaluate it**:

```bash
TM_MODE=eval TM_EVAL_AGENT=sac:tmp/sac docker compose up --build
```

Evaluation drives `sac:` networks without exploration noise (a `decide()` file drives however it's written), and records a run of kind `eval`.

## Compare race results, not reward

Reward depends on the constants in `env.py`. If you change them, a "score" of
500 means something different. The framework ranks everything as a race:

1. finishing beats not finishing
2. among finishes, faster is better
3. among non-finishes, further is better

So `compare_runs.py` works across runs with different rewards.

## Repeatable results: seeds and determinism

- **The game is deterministic.** The same keys give the same result.
- **A driver with no randomness** (rule-based drivers, `sac:` in eval, `linear_agent.py`)
  gets the same result in every episode. `summary.json` reports
  `distinct_outcomes`, which should be **1** for these. If it isn't, something is
  wrong (see [page 9](09-troubleshooting.md)), and that's worth knowing before
  trusting any comparison.
- **Anything random** (exploration, network starting weights, `random`) uses a
  **seed**. It's picked at random and saved in `manifest.json`, or you can set it:
  `TM_SEED=42`. The same seed with the same code and settings repeats the run.
- **Multi-instance training is only roughly repeatable.** With several game
  instances, the order their steps reach the learner varies a little each time.
  The seed is still recorded.

## Luck and seeds

A learning method's result depends partly on luck: the starting weights and
which actions it happened to try. A single run with a different seed can come
out noticeably better or worse.

So when comparing learning methods, consider running each **several times with
different seeds** (for example `TM_SEED=1`, `2`, `3`) and comparing the
spread, not single runs. In `summary.json`, `track_pct.ci95` gives a rough
"plus or minus" on the average for one run's episodes. If two averages are closer
together than their ± ranges, you can't really tell them apart yet.

## Always have baselines

Before celebrating, compare against simple drivers:

```bash
TM_MODE=eval TM_EVAL_AGENT=forward ...                                   # does nothing clever
TM_MODE=eval TM_EVAL_AGENT=agent:examples/wall_follower_agent.py ...     # simple rules
```

If a learned driver can't beat the wall follower, that tells you something.

## Check the run was clean

In `summary.json`:
- `status` should be `completed`
- `step_gaps` should be `0`
- `distinct_outcomes` should be `1` for a deterministic driver

`verify_replay.py` ([page 6](06-results-and-replays.md#checking-a-replay-reproduces))
is a strong check: if a replay reproduces exactly, the whole setup is behaving.

## Keep notes

The manifest records *what* you ran. It can't record *why*. A line in a notebook
("trying a bigger brake margin because it hits the wall at the 2nd corner")
makes your results much easier to understand later.

## Where to go next

- Back to the [index](README.md)
