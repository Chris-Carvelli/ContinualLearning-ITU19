"""
An example of how to use the choice environment
"""

import gym
import custom_envs
import random
import random

from custom_envs.envs.choice import ChoiceEnv
from tests.minigrid.utils import simulate

# env_name = "MiniGrid-Choice3x1-color0-v0"  # goal is green
env_name = "MiniGrid-Choice3x1-color1-v0"  # goal is blue

env = gym.make(env_name)

# # alternatively envs can be created manually with custom size, random starting position, and a custom number of
# # maximum steps before done (otherwise max_step = 2 * (width + height).
# from custom_envs.envs.choice import ChoiceEnv
# random.seed(1)
# env = ChoiceEnv(1, width=5, height=5, random_positions=True, max_steps=10, maze_env=True, euclid_dist_reward=True)
env = ChoiceEnv(1, width=5, height=5, random_positions=True, max_steps=100, maze_env=True, euclid_dist_reward=True)

# show env
while True:
    simulate(env, fps=12)
