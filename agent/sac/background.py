"""Runs SAC learning in its own process, so training and the games never compete
for the GIL.

The games' process keeps a SharedPolicy (a CPU copy of the actor, for choosing
actions) and stores transitions in a SharedReplayBuffer. The learner process
owns the networks and the GPU: it samples that buffer and regularly copies the
actor's weights into the SharedPolicy. All saving happens in the learner, since
it's the only process with the full set of networks.

Keeps to `updates_per_step` gradient updates per stored transition. If
transitions arrive faster than updates can run, the games keep going and the
ratio drops rather than the games slowing down; `updates` lets the trainer
report the ratio it actually achieved.
"""

import queue
import signal
import time

import numpy as np
import torch
import torch.multiprocessing as mp

import best_run

from .agent import SACAgent
from .networks import ActorNetwork

# Updates between copying the actor to the games: frequent enough that they
# drive a recent policy, without moving weights off the GPU on every update.
PUBLISH_EVERY_UPDATES = 20
READY_TIMEOUT_S = 120
STOP_TIMEOUT_S = 15


class SharedPolicy:
    """The games' CPU copy of the actor. Its weights live in shared memory."""

    def __init__(self, input_dims, n_actions, max_action, ctx, chkpt_dir="tmp/sac"):
        self.actor = ActorNetwork(0.0, input_dims, max_action, n_actions=n_actions, chkpt_dir=chkpt_dir, device="cpu")
        self.actor.share_memory()
        self.lock = ctx.Lock()

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float32)
        with torch.no_grad(), self.lock:
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.numpy()[0]


class LearnerProcess:
    def __init__(self, ctx, memory, policy, config):
        self.updates_counter = ctx.RawValue("Q", 0)
        self._commands = ctx.Queue()
        ready = ctx.Event()
        self._process = ctx.Process(
            target=_learner_main,
            args=(memory, policy.actor.state_dict(), policy.lock, self.updates_counter, self._commands, ready, config),
            name="sac-learner",
            daemon=True,
        )
        self._process.start()
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not ready.wait(timeout=1):
            if not self._process.is_alive() or time.monotonic() > deadline:
                self._process.terminate()
                raise RuntimeError("Learner process failed to start")

    @property
    def updates(self):
        return self.updates_counter.value

    def is_alive(self):
        return self._process.is_alive()

    def save_best(self, avg_score):
        self._commands.put(("save_best", avg_score))

    def save_snapshot(self):
        self._commands.put(("snapshot", None))

    def stop(self):
        """Ask the learner to save a snapshot and exit, then wait for it."""
        if self._process.is_alive():
            self._commands.put(("stop", None))
            self._process.join(timeout=STOP_TIMEOUT_S)
        if self._process.is_alive():
            self._process.terminate()


def get_context():
    # spawn, not fork: CUDA can't be used in a forked child.
    return mp.get_context("spawn")


def _publish(actor, shared_state, policy_lock):
    # Off the GPU first, so the games aren't held up by the transfer.
    weights = {name: value.cpu() for name, value in actor.state_dict().items()}
    with policy_lock:
        for name, value in weights.items():
            shared_state[name].copy_(value)


def _learner_main(memory, shared_state, policy_lock, updates, commands, ready, config):
    # Ctrl-C reaches every process in the group. Keep going until the trainer
    # says to stop, so the final snapshot still gets saved.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    torch.set_num_threads(config["torch_threads"])

    agent = SACAgent(
        input_dims=config["input_dims"],
        n_actions=config["n_actions"],
        max_action=config["max_action"],
        chkpt_dir=config["chkpt_dir"],
        memory=memory,
    )
    if config["load_dir"] is not None:
        agent.load_models(config["load_dir"])
    _publish(agent.actor, shared_state, policy_lock)
    ready.set()

    while True:
        try:
            command, value = commands.get_nowait()
        except queue.Empty:
            command = None

        if command == "save_best":
            agent.save_models()
            best_run.save_score(config["best_score_file"], "avg_score", value)
        elif command == "snapshot":
            agent.save_snapshot(config["snapshot_dir"])
        elif command == "stop":
            agent.save_snapshot(config["snapshot_dir"])
            return

        count = memory.mem_cntr
        if count < agent.batch_size or updates.value >= config["updates_per_step"] * count:
            time.sleep(0.001)
            continue

        agent.learn()
        updates.value += 1
        if updates.value % PUBLISH_EVERY_UPDATES == 0:
            _publish(agent.actor, shared_state, policy_lock)
