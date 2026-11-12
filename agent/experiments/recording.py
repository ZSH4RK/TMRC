"""RecordingEnv: wrap a TMNFEnv so every episode driven through it is recorded.

It's a drop-in replacement: same reset()/step()/step_inputs()/close(), and any
other attribute falls through to the wrapped env. Whatever drives it (a SAC
trainer, an evaluation, a competition entry's decide()) gets the same episode
rows and replay files without any recording code of its own.
"""

import time

from .run import Run


class RecordingEnv:
    def __init__(self, env, run: Run, instance: int = 0, policy: str = "", env_config: dict = None):
        self.env = env
        self.run = run
        self.instance = instance
        self.policy = policy
        self.env_config = env_config or {}
        self.recorder = run.recorder(instance=instance, policy=policy, env=self.env_config)
        self._obs = None
        self._episode_started = None
        self.last_row = None

    def __getattr__(self, name):
        # Only called for attributes not found on the wrapper itself.
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        state = self.env.last_game_state
        self.recorder.start(state["position"] if state else None, self.env.last_race_time or 0)
        self._obs = obs
        self._episode_started = time.monotonic()
        return obs

    def step(self, action):
        return self._record(self.env.step(action), action=action)

    def step_inputs(self, inputs):
        return self._record(self.env.step_inputs(inputs), action=None)

    def _record(self, result, action):
        obs, reward, done, info = result
        self.recorder.add(
            input_time_ms=info["input_time_ms"], race_time_ms=info["race_time_ms"], inputs=info["inputs"],
            position=info["position"], velocity=info["velocity"], direction=info["direction"],
            reward=reward, action=action, obs=self._obs, agent_ms=info.get("agent_ms"),
            step_gap=info.get("step_gap", False),
        )
        self._obs = obs
        if done:
            self.last_row = self.end_episode(info)
            info["episode_row"] = self.last_row
        return obs, reward, done, info

    def end_episode(self, info: dict) -> dict:
        """Log the current episode. Called automatically when the env reports
        done; call it yourself for an episode cut short some other way."""
        row = {
            "instance": self.instance,
            "policy": self.policy,
            "done_reason": info.get("done_reason"),
            "race_time_ms": info.get("race_time_ms"),
            "distance_m": round(info["distance_m"], 3) if info.get("distance_m") is not None else None,
            "track_pct": round(info["track_pct"], 3) if info.get("track_pct") is not None else None,
            "wall_s": round(time.monotonic() - self._episode_started, 2) if self._episode_started else None,
            **self.recorder.episode_stats(),
        }
        replay = self.recorder.replay() if len(self.recorder) else None
        return self.run.log_episode(row, replay)

    def close(self):
        self.env.close()
