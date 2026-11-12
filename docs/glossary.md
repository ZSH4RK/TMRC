# Glossary

**Action:** what a driver chooses each step. Either four keys (left, right,
accelerate, brake), or two numbers (steer, throttle) that get turned into keys.

**Agent / driver:** the code that decides what to press. In this project, often
a file with a `decide()` function.

**Baseline:** a simple driver to compare against, like `forward` or the wall follower.

**Checkpoint:** a saved copy of a neural network's weights, so training can stop and resume.

**Confidence interval (ci95):** a "plus or minus" range around an average. Roughly:
if you ran again, the average would usually land within it.

**Container / image (Docker):** an *image* is a packaged setup (game, Wine,
Python, code). A *container* is a running copy of it.

**Deterministic:** gives the same result every time from the same starting point.
TrackMania's physics are deterministic.

**Episode:** one attempt at the track, from the start until finish, timeout or stuck.

**Evaluation (eval):** testing a frozen driver without exploration, to measure how good it is.

**Exploration:** a learning driver trying actions it's unsure about, to discover better ones.

**Hash:** a short fingerprint of a file's contents. Change one character and the hash changes.
Used to tell whether two runs used the same code.

**Hill climbing:** improve something by making small random changes and keeping the ones that help.

**Instance:** one running copy of the game. Several can run at once.

**Manifest:** `manifest.json` in a run folder: everything about how the run was set up.

**noVNC:** shows the game's screen in your web browser.

**Observation (obs):** the numbers a driver gets each step, describing what it can "see".

**Parameters / weights:** the adjustable numbers inside a driver that learning changes.

**Policy:** the rule (often a neural network) that maps observations to actions.

**Ray:** an invisible line from the car, used to measure the distance to a wall in one direction.

**Reinforcement learning (RL):** learning to act by trial and error, guided by rewards.

**Replay:** a recording of every step of an episode: keys, timing and route.

**Reward:** a number given each step saying how good it was. RL tries to maximise the total.

**Run:** one execution of anything (training, eval, …), recorded as a folder in `results/`.

**SAC (Soft Actor-Critic):** the reinforcement learning algorithm included in this project.

**Seed:** a number that fixes the sequence of "random" numbers, making randomness repeatable.

**Step:** one decision: 50ms of game time by default.

**Step gap:** game time jumping by more than one step between decisions. It shouldn't happen.

**TMInterface:** a TrackMania mod that lets programs read the game's state and press keys.

**Wine:** software that runs Windows programs on Linux.
