import sys

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
        self.i = 1
        self.reset()

    def target_index(self):
        return self.i - (self.length + 2)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        reward = 0
        if self.i >= len(self.obs):
            return np.zeros(self.height + 2), reward, True, dict()
        obs = self.obs[self.i]
        if 0 <= self.target_index() < self.length:
            p = 1
            target = self.obs[self.target_index() + 1][2:]
            match = np.sum(1 - np.abs(action - target) ** p) / self.height
            # min_match = 1 - .5**p
            min_match = 0.25
            if match > min_match:
                reward = (match - min_match) / ((1 - min_match) * self.length)
            # print(match, action, target)
        done = self.i >= len(self.obs) - 1
        self.i += 1
        return obs, reward, done, dict()

    def reset(self):
        bits = np.random.randint(0, 2, (self.length, self.height))
        extra = np.zeros((self.length, 2))
        self.obs = np.concatenate((extra, bits), 1)
        self.obs = np.concatenate(
            (np.zeros((1, self.height + 2)), self.obs, np.zeros((self.length + 2, self.height + 2))), 0)
        self.obs[0][0] = 1
        self.obs[self.length + 1][1] = 1
        self.i = 1
        # print(bits)
        return self.obs[0]

    def render(self, mode='human'):
        print("No rendering for copy env yet")

    def seed(self, seed=None):
        np.random.seed(seed)


class RandomCopy(Copy):
    """Creates a copy env of varying length whenever reset is called"""

    def __init__(self, height=3, min_length=1, max_length=12):
        self.max_length = max_length
        self.min_length = min_length
        super().__init__(height=height, length=-1)

    def reset(self):
        self.length = int(np.random.randint(self.min_length, self.max_length + 1))
        return super().reset()


class PerfectModel:
    def __init__(self, env):
        self.env = env
        self.i = -1
        self.outputs = np.concatenate((self.env.obs[self.env.length + 2:], self.env.obs[1:self.env.length + 2]), 0)

    def __call__(self, *args, **kwargs):
        self.i += 1
        return self.outputs[self.i % len(self.outputs)][2:]

    def obs_to_input(self, obs):
        return obs

    def get_action(self, y, env):
        return y

    def reset(self):
        self.__init__(self.env)


class ImperfectModel:
    def __init__(self, env):
        self.env = env

    def __call__(self, *args, **kwargs):
        return np.zeros(self.env.height) + 0

    def obs_to_input(self, obs):
        return obs

    def get_action(self, y, env):
        return y

    def reset(self):
        self.__init__(self.env)


def test_randomness():
    h = 1
    l = 4
    s = 0
    for i in range(10000):
        bits = np.random.randint(0, 2, (h, l))
        s += np.average(bits)
    print(s)

if __name__ == '__main__':
    # test_randomness()
    # target = np.array([1,0,1,0,1])
    # action = np.array([0,0,0,0,0])
    # action = np.zeros(len(target)) + 0.5
    # # action[0:1] = [1]
    # print(action)
    # # print(np.abs(target - action)** 3.1)
    # match = np.sum(1 - np.abs(action - target)**2) / len(action)
    # print(match)

    #
    copy_size = 2
    length = 2
    # c = Copy(copy_size, 4)

    # c = gym.make(f"Copy-{copy_size}x{length}-v0")
    c = gym.make(f"CopyRnd-{copy_size}-v0")
    c.reset()
    # print(c)
    # s = c.reset()

    # inputs = np.concatenate((c.obs[c.length + 1:], c.obs[:c.length + 1]), 0)
    # print(inputs.transpose())
    # for i, d in enumerate(inputs):
    #     step = c.step(d[2:])
    #     print(i, d[2:], step[0:3])

    # net = CopyNTM(copy_size, 22)
    # net.history = defaultdict(list)
    net = PerfectModel(c)
    # net = ImperfectModel(c)
    s = 0
    n = 50
    for x in range(n):
        res = evaluate_model(c, net, 100000, n=1)
        print(res)
        s += res[0]
    print(s / n)
    # print(net.inputs)
    # net.plot_history()

    #
    # print(inputs.transpose())
    # n = int(len(c.obs) / 2) + 1
    # for i in range(2 * n):
    #     step = c.step(c.obs[i % n][2:])
    #     print(i, c.obs[i % n][2:], step[0:3])
