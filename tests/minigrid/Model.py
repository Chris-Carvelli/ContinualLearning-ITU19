import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gym

import time


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        # self.actor = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 7)
        # )

        self.out = nn.Linear(64, 4)

        self.add_tensors = {}
        self.init()

    def forward(self, x):
        # x = x.reshape([1, 147])
        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        return self.out(x)

    def evolve(self, sigma):
        params = self.named_parameters()
        for name, tensor in sorted(params):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                tensor.data.normal_(0, 1)
                tensor.data *= 1 / torch.sqrt(tensor.pow(2).sum(1, keepdim=True))
            else:
                tensor.data.zero_()


def evaluate_model(env_key, model, max_eval, render=False, fps=60):
    env = gym.make(env_key)
    # env = FlatObsWrapper(env)
    state = env.reset()

    model.eval()

    tot_reward = 0
    reward = 0
    n_eval = 0
    # TMP
    action_freq = np.zeros([7])
    while reward == 0 and n_eval < max_eval:
        state = state['image']

        # removed some scaffolding, check if something was needed
        values = model(Variable(torch.Tensor([state])))
        # values = env.step(env.action_space.sample())
        action = np.argmax(values.data.numpy()[:env.action_space.n])

        # TMP remapping toggle action
        if action is 3:
            action = 5

        action_freq[action] += 1
        state, reward, is_done, _ = env.step(action)
        if render:
            print('hello')
            env.render('human')
            print('action=%s, reward=%.2f' % (action, reward))
            time.sleep(1/fps)

        tot_reward += reward
        n_eval += 1

    env.close()
    if tot_reward > 0:
        print(f'action_freq: {action_freq/n_eval}\treward: {tot_reward}')
    return tot_reward
