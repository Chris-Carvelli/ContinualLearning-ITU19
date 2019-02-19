import numpy
import time

import gym
import numpy as np
import torch

from models.ntm import CopyNTM, evaluate_model
from collections import defaultdict
from custom_envs import *


class Copy(gym.Env):
    # Set this in SOME subclasses
    # metadata = {'render.modes': []}
    # spec = None
    reward_range = (0, 1)

    action_space = None
    observation_space = None

    def __init__(self, height=3, length=3):
        self.action_space = gym.spaces.box.Box(0, 1, [height], dtype=np.float32)
        self.observation_space = gym.spaces.box.Box(0, 1, [height + 2], dtype=np.float32)
        super().__init__()
        self.height = height
        self.length = length
        # self.bits = None
        self.obs = None
        self.i = -1
        self.reset()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        self.i += 1
        done = self.i >= len(self.obs)
        reward = 0
        if done:
            return np.zeros(self.height + 2), reward, done, dict()
        obs = self.obs[self.i]
        if self.length + 2 <= self.i < 2 * self.length + 2:
            # print(action, self.obs[self.i][2:])
            reward = np.sum(0.5 - np.abs(action - self.obs[self.i -(self.length + 2)][2:])) / (self.height * self.length)
        return obs, reward, done, dict()

    def reset(self):
        bits = np.random.randint(0, 2, (self.length, self.height))
        extra = np.zeros((self.length, 2))
        self.obs = np.concatenate((extra, bits), 1)
        self.obs = np.concatenate(
            (np.zeros((1, self.height + 2)), self.obs, np.zeros((self.length + 1, self.height + 2))), 0)
        self.obs[0][0] = 1
        self.obs[self.length + 1][1] = 1
        self.i = 0
        return self.obs[self.i]

    def render(self, mode='human'):
        print("No rendering for copy env yet")

    def seed(self, seed=None):
        np.random.seed(seed)


if __name__ == '__main__':
    copy_size = 4
    length = 4
    # c = Copy(copy_size, 4)

    c = gym.make(f"Copy-{copy_size}x{length}-v0")
    # print(c)
    s = c.reset()

    # inputs = np.concatenate((c.obs[c.length + 1:], c.obs[:c.length + 1]), 0)
    # print(inputs.transpose())
    # for i, d in enumerate(inputs):
    #     step = c.step(d[2:])
    #     print(i, d[2:], step[0:3])

    net = CopyNTM(copy_size, max_memory=copy_size + 2)
    net.history = defaultdict(list)
    res = evaluate_model(c, net, 100000)
    print(res)
    net.plot_history()





    #
    # print(inputs.transpose())
    # n = int(len(c.obs) / 2) + 1
    # for i in range(2 * n):
    #     step = c.step(c.obs[i % n][2:])
    #     print(i, c.obs[i % n][2:], step[0:3])
