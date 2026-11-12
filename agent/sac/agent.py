"""SAC agent. Ported from Soft-Actor-Critic/sac_torch.py.

Same algorithm (twin Q critics + a value/target-value network updated by
Polyak averaging, as in Tabor's SAC implementation) — the only real change is
that `max_action` is passed in directly instead of read off a gym
`env.action_space`, since TMNFEnv (env.py) isn't a gym env.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork, ValueNetwork


class SACAgent:
    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[14],
        max_action=None,
        gamma=0.99,
        tau=0.005,
        n_actions=2,
        max_size=100_000,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
        chkpt_dir="tmp/sac",
        memory=None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = memory if memory is not None else ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        if max_action is None:
            max_action = [1.0] * n_actions

        self.actor = ActorNetwork(alpha, input_dims, max_action, n_actions=n_actions, name="actor", chkpt_dir=chkpt_dir)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name="critic1", chkpt_dir=chkpt_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name="critic2", chkpt_dir=chkpt_dir)

        self.value = ValueNetwork(beta, input_dims, name="value", chkpt_dir=chkpt_dir)
        self.target_value = ValueNetwork(beta, input_dims, name="target_value", chkpt_dir=chkpt_dir)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # np.array(...) first: torch.tensor([observation], ...) on a single
        # ndarray wrapped in a list forces a slow per-element conversion.
        state = torch.tensor(np.array([observation]), dtype=torch.float32).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def choose_deterministic_action(self, observation):
        """Mean action, no exploration noise — used at inference time by my_agent.py."""
        state = torch.tensor(np.array([observation]), dtype=torch.float32).to(self.actor.device)
        action = self.actor.deterministic_action(state)

        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_values_state_dict = dict(target_value_params)
        values_state_dict = dict(value_params)

        for name in values_state_dict:
            values_state_dict[name] = tau * values_state_dict[name].clone() + (1 - tau) * target_values_state_dict[name].clone()

        self.target_value.load_state_dict(values_state_dict)

    def _networks(self):
        return (self.actor, self.value, self.target_value, self.critic_1, self.critic_2)

    def save_models(self):
        print(".....Saving Models.....")

        for network in self._networks():
            network.save_checkpoint()

    def load_models(self, checkpoint_dir=None):
        """Load from each network's own chkpt_dir, or from checkpoint_dir if given."""
        print(".....Loading Models.....")

        for network in self._networks():
            if checkpoint_dir is None:
                network.load_checkpoint()
            else:
                path = os.path.join(checkpoint_dir, network.name + "_sac")
                network.load_state_dict(torch.load(path, map_location=network.device))

    def save_snapshot(self, checkpoint_dir):
        """Save the current networks to checkpoint_dir, whatever their score, so
        a stopped run can resume without touching the best-scoring checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        for network in self._networks():
            torch.save(network.state_dict(), os.path.join(checkpoint_dir, network.name + "_sac"))

    def learn(self):
        """One gradient update. Returns its training stats (see the end), or
        None if the buffer doesn't hold a batch yet."""
        if self.memory.mem_cntr < self.batch_size:
            return None

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        # Detached tensors, not floats: .item() would sync with the GPU on
        # every update. Callers that log them sum and convert in batches.
        return {
            "value_loss": value_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": -log_probs.detach().mean(),
            "q_mean": critic_value.detach().mean(),
            "batch_reward_mean": reward.mean(),
        }
