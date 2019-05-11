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


class ExplorationBonusIfZero(gym.core.Wrapper):
    """
    Wrapper which adds a small exploration bonus for each new state (pos, dirm, action), but only if the env finds no
    rewards
    """
    expl_reward_value = 0.001

    def __init__(self, env):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        self._reset_exploration()
        super().__init__(env)

    def _reset_exploration(self):
        self._explored_states = set()
        self.got_rewards = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if not self.got_rewards:
            env = self.unwrapped
            tup = (tuple(env.agent_pos), env.agent_dir, action)

            self.got_rewards = reward != 0
            if self.got_rewards:
                # Subtract all exploration rewards given already
                reward -= len(self._explored_states) * self.expl_reward_value
            elif tup not in self._explored_states:
                reward += self.expl_reward_value
            self._explored_states.add(tup)

        return obs, reward, done, info

    def reset(self, **kwargs):
        self._reset_exploration()
        return self.env.reset(**kwargs)
