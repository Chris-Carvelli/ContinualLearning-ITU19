import math

import gym


class ExplorationBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.1
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        super().__init__(env)
        self.explored_states = set()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        if tup not in self.explored_states:
            reward += 0.001
            self.explored_states.add(tup)

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.explored_states = set()
        return self.env.reset(**kwargs)