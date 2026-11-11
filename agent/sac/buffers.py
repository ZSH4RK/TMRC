"""Replay buffers for SAC. ReplayBuffer is ported from Soft-Actor-Critic/buffers.py,
unchanged. SharedReplayBuffer lets train_sac_vec.py's games store transitions
while a separate learner process samples them.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class SharedReplayBuffer(ReplayBuffer):
    """A ReplayBuffer held in shared memory, safe to use from several processes.

    Pass it as a Process argument; the child attaches to the same memory.
    """

    def __init__(self, max_size, input_shape, n_actions, ctx):
        self.mem_size = max_size
        self._input_shape = tuple(input_shape)
        self._n_actions = n_actions
        self._lock = ctx.Lock()
        self._counter = ctx.RawValue("Q", 0)
        state_size = int(np.prod(input_shape))
        self._raw = {
            "state": ctx.RawArray("f", max_size * state_size),
            "new_state": ctx.RawArray("f", max_size * state_size),
            "action": ctx.RawArray("f", max_size * n_actions),
            "reward": ctx.RawArray("f", max_size),
            "terminal": ctx.RawArray("b", max_size),
        }
        self._attach()

    def _attach(self):
        states = (self.mem_size, *self._input_shape)
        self.state_memory = np.frombuffer(self._raw["state"], dtype=np.float32).reshape(states)
        self.new_state_memory = np.frombuffer(self._raw["new_state"], dtype=np.float32).reshape(states)
        self.action_memory = np.frombuffer(self._raw["action"], dtype=np.float32).reshape(self.mem_size, self._n_actions)
        self.reward_memory = np.frombuffer(self._raw["reward"], dtype=np.float32)
        self.terminal_memory = np.frombuffer(self._raw["terminal"], dtype=bool)

    def __getstate__(self):
        # The numpy views can't cross processes; the child rebuilds them.
        names = ("mem_size", "_input_shape", "_n_actions", "_lock", "_counter", "_raw")
        return {name: self.__dict__[name] for name in names}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._attach()

    @property
    def mem_cntr(self):
        return self._counter.value

    @mem_cntr.setter
    def mem_cntr(self, value):
        self._counter.value = value

    def store_transition(self, state, action, reward, state_, done):
        with self._lock:
            super().store_transition(state, action, reward, state_, done)

    def sample_buffer(self, batch_size):
        with self._lock:
            return super().sample_buffer(batch_size)
