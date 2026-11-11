"""Soft Actor-Critic, ported from Soft-Actor-Critic/ (sac_torch.py, networks.py, buffers.py).

- `sac.buffers.ReplayBuffer` — experience replay
- `sac.networks` — ActorNetwork / CriticNetwork / ValueNetwork
- `sac.agent.SACAgent` — ties the above together (choose_action, remember, learn)
"""
