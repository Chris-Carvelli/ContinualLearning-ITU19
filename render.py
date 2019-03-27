from test import *
from custom_envs import *
import gym_minigrid
import dill
import gym
from gym.wrappers import Monitor

from custom_envs.envs.t_maze import TMaze

fp = open('res.dill',  'rb')
res = None
while True:
    try:
        res = dill.load(fp)
    except EOFError:
        fp.close()
        break

env_key = 'TMaze-1x10-v0'
# env = gym.make(env_key)
# env_key = 'BipedalWalker-v2'
env = Monitor(gym.make(env_key), 'render', force=True)
# res[-1][0][0].evaluate(env, 100000, True)
env.reset()
env.render()
env.close()
