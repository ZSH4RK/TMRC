# TMRC documentation: start here

TMRC lets you build a program that drives a car in **TrackMania Nations Forever**
and measure how well it does. The game runs inside a Docker container. Your code
gets told where the car is, decides which keys to press, and the framework
records every run so results can be compared and repeated.

There is **no single right way** to build a driver. You could write rules by
hand, tune numbers by trial and error, use reinforcement learning, copy a human,
or invent something new. The framework is the same for all of them. These docs
explain the tools; what you build with them is up to you.

---

## Which page should I read?

Everyone should read these two first:

| Page | What you'll learn |
|---|---|
| [1. The big picture](01-big-picture.md) | How the pieces fit together, in plain language |
| [2. Setup and your first run](02-setup-and-first-run.md) | Getting it running and watching the car drive |

Then pick whatever matches what you want to do. They don't need to be read in order.

| I want to… | Read |
|---|---|
| Write a driver myself, with my own logic | [3. Writing an agent](03-writing-an-agent.md) |
| Write code that *learns* to drive, with any method | [4. The environment](04-the-environment.md), then [5. Ways to learn](05-ways-to-learn.md) |
| Understand reinforcement learning and the included SAC trainer | [5. Ways to learn](05-ways-to-learn.md) |
| Find out how well something did, or watch a replay | [6. Results and replays](06-results-and-replays.md) |
| Compare two ideas fairly and know which is really better | [7. Running fair experiments](07-fair-experiments.md) |
| Look up a setting | [8. Settings reference](08-settings-reference.md) |
| Fix something that isn't working | [9. Troubleshooting](09-troubleshooting.md) |
| Change or extend the framework itself | [10. How the framework works](10-how-the-framework-works.md) |
| Look up a word | [Glossary](glossary.md) |

## Some possible paths

These are examples, not instructions. Mix them, skip them, or ignore them.

- **"I'm new to coding."** Read 1 and 2, then open `agent/examples/forward_agent.py`
  (8 lines) and `agent/examples/wall_follower_agent.py`. Change a number in the
  wall follower, evaluate it ([page 3](03-writing-an-agent.md)), and see what changes.
- **"I know Python, not machine learning."** Write a rule-based driver
  ([page 3](03-writing-an-agent.md)), then try improving its numbers
  automatically with `agent/examples/hill_climb.py` ([page 5](05-ways-to-learn.md)).
- **"I want to do reinforcement learning."** Read [page 4](04-the-environment.md)
  and [page 5](05-ways-to-learn.md). Run the SAC trainer as it is, or write your own
  algorithm against the same environment.
- **"I want to research what works."** Read [page 7](07-fair-experiments.md) first,
  then use whichever approach you're testing.

## Example files

| File | What it is |
|---|---|
| `agent/examples/forward_agent.py` | The simplest possible driver: always accelerate |
| `agent/examples/wall_follower_agent.py` | A hand-written rule-based driver |
| `agent/examples/hill_climb.py` | A complete learning loop you write yourself (trial-and-error improvement) |
| `agent/examples/linear_agent.py` | Drives with the numbers `hill_climb.py` learned |
| `agent/my_agent.py` | Drives with the neural network the SAC trainer learned |
| `agent/train_sac_vec.py` | The included reinforcement learning trainer |
