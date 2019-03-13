import gym
from typing import List


class MultiEnv(gym.Env):

    def __init__(self, envs: List[gym.Env], rounds=1, close_prevous_env=False):
        self.close_prevous_env = close_prevous_env
        self.schedule: List[(gym.Env, int)] = [(env, rounds) for env in envs]
        self.i: int = None
        self.env: gym.Env = None
        self.rewards: List[float] = None
        self.total_rounds = len(envs) * rounds
        self.round: int = None
        self.to_close_list: List[gym.Env] = []   # List of previous env to close before rendering next
        self.reset()

    def _set_env(self, env):
        """Sets the current env"""
        self.env = env
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.on_env_change()

    def _next_round(self):
        self.round += 1
        env_change = False
        done = False
        if self.round >= self.schedule[self.i][1]:
            self.round = 0
            self.i += 1
            if self.i < len(self.schedule):
                if self.close_prevous_env:
                    self.to_close_list.append(self.env)
                self.env = self.schedule[self.i][0]
                env_change = True
                self.on_env_change()
            else:
                done = True
        if not done:
            self.on_new_round(env_change)
        state = self.env.reset()
        return done, state

    def step(self, action):
        done = False
        state, reward, round_done, info = self.env.step(action)
        self.rewards.append(reward)
        if round_done:
            done, state = self._next_round()
        obs = self._get_obs(state, round_done, reward)
        score = reward / max(2, self.total_rounds)
        return obs, score, done, info

    def reset(self):
        self.i: int = 0
        self.env: gym.Env = self.schedule[self.i][0]
        self.rewards = []
        self.round: int = 0
        state = self.env.reset()
        obs = self._get_obs(state)
        return obs

    def _get_obs(self, state, round_done=False, reward=0):
        obs = dict()
        if isinstance(state, dict):
            assert "round_done" not in state
            assert "reward" not in state
            assert "env" not in state
            obs = state
        else:
            obs["state"] = state
        obs["round_done"] = round_done
        obs["reward"] = reward
        obs["env"] = type(self.env)
        return obs

    def render(self, mode='human', **kwargs):
        for e in self.to_close_list:
            e.close()
        self.to_close_list = []
        self.env.render(mode, **kwargs)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        raise NotImplementedError

    def __str__(self):
        if self.spec is None:
            info = str([f"{n}: {type(env).__name__}" for env, n in self.schedule])
            return f'<{type(self).__name__} instance with {info}>'
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def on_env_change(self):
        """Override to implement specific behaviour when env change"""
        pass

    def on_new_round(self, env_change: bool):
        """Override to implement specific behaviour when one new round change"""
        pass
