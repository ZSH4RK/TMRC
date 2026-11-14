"""SAC agent. Ported from Soft-Actor-Critic/sac_torch.py.

Same algorithm (twin Q critics + a value/target-value network updated by
Polyak averaging, as in Tabor's SAC implementation) — the only real change is
that `max_action` is passed in directly instead of read off a gym
`env.action_space`, since TMNFEnv (env.py) isn't a gym env.

Two optional additions, both off by default so the class alone behaves like the
original (the trainers turn them on through sac/settings.py):

- auto_entropy: learn the entropy temperature (Haarnoja et al. 2018, "SAC
  Algorithms and Applications") so the policy's entropy tracks target_entropy.
  The original uses a fixed temperature of 1 against reward * reward_scale,
  which lets the policy grow more random again once it drives well.
- lr_decay_updates: fall the learning rates linearly to lr_final of their
  starting values over that many updates, so a good policy settles instead of
  being pushed around by full-size updates forever.

The temperature and update count are saved beside the networks (sac_state), so
a resumed run carries on from them.
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
        lr_decay_updates=0,
        lr_final=0.1,
        auto_entropy=False,
        target_entropy=None,
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

        self.chkpt_dir = chkpt_dir
        self.updates = 0
        self.lr_decay_updates = lr_decay_updates
        self.lr_final = lr_final
        self._base_lrs = [(net.optimizer, lr) for net, lr in
                          ((self.actor, alpha), (self.critic_1, beta), (self.critic_2, beta), (self.value, beta))]
        self.auto_entropy = auto_entropy
        self.target_entropy = -float(n_actions) if target_entropy is None else float(target_entropy)
        # log of the temperature; 0 = temperature 1, the original's fixed value.
        self.log_temperature = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=alpha)

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

    def _state(self):
        return {"log_temperature": self.log_temperature.detach().cpu(), "updates": self.updates}

    def _load_state(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "sac_state")
        if not os.path.exists(path):
            return  # a checkpoint from before sac_state existed
        state = torch.load(path, map_location="cpu")
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.log_temperature.device))
        self.updates = int(state["updates"])

    def save_models(self):
        print(".....Saving Models.....")

        for network in self._networks():
            network.save_checkpoint()
        torch.save(self._state(), os.path.join(self.chkpt_dir, "sac_state"))

    def load_models(self, checkpoint_dir=None):
        """Load from each network's own chkpt_dir, or from checkpoint_dir if given."""
        print(".....Loading Models.....")

        for network in self._networks():
            if checkpoint_dir is None:
                network.load_checkpoint()
            else:
                path = os.path.join(checkpoint_dir, network.name + "_sac")
                network.load_state_dict(torch.load(path, map_location=network.device))
        self._load_state(checkpoint_dir or self.chkpt_dir)

    def save_snapshot(self, checkpoint_dir):
        """Save the current networks to checkpoint_dir, whatever their score, so
        a stopped run can resume without touching the best-scoring checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        for network in self._networks():
            torch.save(network.state_dict(), os.path.join(checkpoint_dir, network.name + "_sac"))
        torch.save(self._state(), os.path.join(checkpoint_dir, "sac_state"))

    def learning_rate_fraction(self):
        if self.lr_decay_updates <= 0:
            return 1.0
        progress = min(1.0, self.updates / self.lr_decay_updates)
        return 1.0 - (1.0 - self.lr_final) * progress

    def _apply_lr_decay(self):
        fraction = self.learning_rate_fraction()
        for optimizer, base in self._base_lrs + [(self.temperature_optimizer, self._base_lrs[0][1])]:
            for group in optimizer.param_groups:
                group["lr"] = base * fraction

    def learn(self):
        """One gradient update. Returns its training stats (see the end), or
        None if the buffer doesn't hold a batch yet."""
        if self.memory.mem_cntr < self.batch_size:
            return None

        if self.lr_decay_updates > 0:
            self._apply_lr_decay()
        temperature = self.log_temperature.exp().detach() if self.auto_entropy else 1.0

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
        value_target = critic_value - temperature * log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = temperature * log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        if self.auto_entropy:
            # Raises the temperature when entropy (-log_probs) is below the
            # target, lowers it when above.
            temperature_loss = -(self.log_temperature * (log_probs.detach() + self.target_entropy)).mean()
            self.temperature_optimizer.zero_grad()
            temperature_loss.backward()
            self.temperature_optimizer.step()

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
        self.updates += 1

        # Detached tensors, not floats: .item() would sync with the GPU on
        # every update. Callers that log them sum and convert in batches.
        return {
            "value_loss": value_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": -log_probs.detach().mean(),
            "q_mean": critic_value.detach().mean(),
            "batch_reward_mean": reward.mean(),
            "temperature": self.log_temperature.detach().exp().squeeze(),
            "lr": torch.tensor(self._base_lrs[0][0].param_groups[0]["lr"]),
        }
