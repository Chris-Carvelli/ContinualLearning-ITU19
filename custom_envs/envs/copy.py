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

    def __init__(self, height=3, length=3, reward_precopy_bits=False):
        self.action_space = gym.spaces.box.Box(0, 1, [height], dtype=np.float32)
        self.observation_space = gym.spaces.box.Box(0, 1, [height + 2], dtype=np.float32)
        super().__init__()
        self.height = height
        self.length = length
        self.obs = None
        self.targets = None
        self.actions = None
        self.reward_precopy_bits = reward_precopy_bits  # if True rewards will be given for all target columns
        self.i = 0
        self.reset()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        self.actions.append(np.copy(action))
        reward = 0
        done = self.i >= len(self.obs) - 1
        if done:
            obs = np.zeros(self.height + 2) - 1
        else:
            obs = self.obs[self.i + 1]
        if self.i >= len(self.targets):
            return obs, reward, done, dict()
        target = self.targets[self.i]
        reward = self._reward(action, target)
        self.i += 1
        return obs, reward, done, dict()

    def _reward(self, action, target):
        if not self.reward_precopy_bits and target[0] == 0.5:
            return 0
        p = 1
        match = np.sum(1 - np.abs(action - target) ** p) / self.height
        # min_match = 1 - .5**p
        min_match = 0.25
        length = self.length
        if self.reward_precopy_bits:
            length = len(self.obs)
        if match > min_match:
            return (match - min_match) / ((1 - min_match) * length)
        return 0

    def reset(self):
        bits = np.random.randint(0, 2, (self.length, self.height))
        extra = np.zeros((self.length, 2))
        self.obs = np.concatenate((extra, bits), 1)
        self.obs = np.concatenate(
            (np.zeros((1, self.height + 2)), self.obs, np.zeros((self.length + 1, self.height + 2))), 0)
        self.obs[0][0] = 1
        self.obs[self.length + 1][1] = 1
        self.targets = np.concatenate((np.zeros((self.length + 2, self.height)) + .5, bits), 0)
        self.actions = []
        self.i = 0
        return self.obs[0]

    def render(self, mode='human'):
        i = self.i - 1
        if i == 0:
            print(f"{'obs':25}|{'target':25}|{'action':25}|{'reward'}")
        n = max(len(self.obs), len(self.targets))
        if 0 <= i < n:
            obs = str(self.obs[i])
            target = str(self.targets[i])
            action = str(np.round(self.actions[i], 2))
            reward = self._reward(self.actions[i], self.targets[i])
            print(f"{obs:25}|{target:25}|{action:25}|{reward:.3f}")
        if i == n - 1:
            r = sum([self._reward(self.actions[i], self.targets[i]) for i in range(n)])
            print(f"accumulated reward = {r:.4f}")

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
        self.i = None
        self.outputs = None
        self.reset()

    def __call__(self, *args, **kwargs):
        self.i += 1
        return self.outputs[self.i]

    def obs_to_input(self, obs):
        return obs

    def get_action(self, y, env):
        return y

    def reset(self):
        self.i = -1

        # start = np.zeros((self.env.length + 2, self.env.height + 2))
        # print(start.shape)
        # print(self.env.obs[1:self.env.length + 1].shape)
        # self.outputs = np.concatenate((start, self.env.obs[1:self.env.length + 1]), 0)
        self.outputs = self.env.targets


class ImperfectModel:
    def __init__(self, env, v=0.5):
        self.env = env
        self.val = v

    def __call__(self, *args, **kwargs):
        return np.full(self.env.height, self.val)

    def obs_to_input(self, obs):
        return obs

    def get_action(self, y, env):
        return y

    def reset(self):
        self.__init__(self.env, self.val)


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
    length = 4
    # c = Copy(copy_size, 4)

    c = gym.make(f"Copy-{copy_size}x{length}-v0")
    # c = gym.make(f"CopyRnd-{copy_size}-v0")
    c.reset()

    # print(c.obs)
    # print(c.targets)
    # c.render()
    # print(c)
    # s = c.reset()

    # inputs = np.concatenate((c.obs[c.length + 1:], c.obs[:c.length + 1]), 0)
    # print(inputs.transpose())
    # for i, d in enumerate(inputs):
    #     step = c.step(d[2:])
    #     print(i, d[2:], step[0:3])

    net = PerfectModel(c)
    # net = ImperfectModel(c, v=1.0)
    # net = CopyNTM(copy_size, 22)
    net.history = defaultdict(list)
    evaluate_model(c, net, 100000, n=1, render=True)

    # s = 0
    # n = 1000
    # for x in range(n):
    #     res = evaluate_model(c, net, 100000, n=1)
    #     # print(res)
    #     s += res[0]
    # print(s / n)
    # print(net.inputs)
    if hasattr(net, "plot_history"):
        net.plot_history()

    #
    # print(inputs.transpose())
    # n = int(len(c.obs) / 2) + 1
    # for i in range(2 * n):
    #     step = c.step(c.obs[i % n][2:])
    #     print(i, c.obs[i % n][2:], step[0:3])

    # print(c.obs)
    # print(net.reset())
    # for i in range(length * 2 + 1):
    #     print(net(0))
