from test import *
# from custom_envs import *
# from gym_minigrid import *
import dill
import gym
from gym.wrappers import Monitor

from custom_envs.envs.t_maze import TMaze, SingleTMaze
from gym_minigrid.envs.empty import *


env_key = 'TMaze-2x10-v0'
# env_key = 'SingleTMaze-v0'
# env_key = 'MiniGrid-Empty-6x6-v0'
# env_key = 'BipedalWalker-v2'
env = gym.make(env_key)
# env = SingleTMaze()
env = Monitor(env, 'render', force=True)
# res[-1][0][0].evaluate(env, 100000, True)
env.reset()
# env.render("rgb_array")
env.render()
env.step(2)
env.render()
env.step(2)
env.render()
env.step(1)
env.render()
env.step(2)
env.render()
env.step(2)
env.render()
env.render()
env.render()
env.close()