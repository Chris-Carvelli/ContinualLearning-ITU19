import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce

import gym
from memory_profiler import profile

import time

from tests.minigrid.utils import random_z_v

# TODO create proper setting file (as .cfg)
Z_DIM = 32
Z_VECT_EVOLUTION_PROBABILITY = 0.5
# TODO compute in HyperNN.__init__()
Z_NUM = 4


class HyperNN(nn.Module):
    def __init__(self, named_parameters=None):
        super().__init__()

        # TODO examine shapes of all layers and get max
        max_size = 32 * 64 * 2 * 2

        # TODO get n layers from len(shapes)
        self.z_v = random_z_v(Z_DIM, Z_NUM)

        self.l1 = nn.Linear(Z_DIM, 128)
        self.l2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, max_size)

        self.add_tensors = {}

        self.init()

    def forward(self, layer_index):
        x = chunks(self.z_v, Z_DIM)[layer_index]

        # x = torch.from_numpy(x).float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        return self.out(x)

    # @profile
    def evolve(self, sigma):
        p = torch.distributions.normal.Normal(0.5, 0.1).sample().item()
        if p > Z_VECT_EVOLUTION_PROBABILITY:
            # evolve z vector
            self.z_v += torch.distributions.normal.Normal(torch.zeros([Z_DIM * Z_NUM]), sigma).sample()
        else:
            # evolve weights
            for name, tensor in self.named_parameters():
                # to_add = self.add_tensors[tensor.size()]
                # to_add.normal_(0.0, sigma)
                tensor.data.add_(torch.distributions.normal.Normal(tensor, 0.1).sample())

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal_(tensor)
            else:
                tensor.data.zero_()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.hyperNN = HyperNN()

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

        self.out = nn.Linear(64, 4)

        self.add_tensors = {}

        self.update_weights()

    def forward(self, x):
        # x = x.reshape([1, 147])
        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        return self.out(x)

    def evolve(self, sigma):
        self.hyperNN.evolve(sigma)
        self.update_weights()

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                tensor.data.normal_(0, 1)
                tensor.data *= 1 / torch.sqrt(tensor.pow(2).sum(1, keepdim=True))
            else:
                tensor.data.zero_()

    def update_weights(self):
        # TODO find better impl
        z_chunk = 0
        for i, layer in enumerate(self.image_conv):
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    self.image_conv[i].weight = self.get_weights(z_chunk, layer.weight.shape)
                    z_chunk += 1

    def get_weights(self, layer_index, layer_shape):
        w = self.hyperNN(layer_index)
        w = torch.narrow(w, 0, 0, reduce((lambda x, y: x * y), layer_shape))
        w = w.view(layer_shape)

        return torch.nn.Parameter(w)

    def become_child(self, parent):
        params = dict(parent.hyperNN.named_parameters())
        for name, param in self.hyperNN.named_parameters():
            if 'weight' in name:
                # TODO check if possible .data = .data
                param.data.copy_(params[name])


def evaluate_model(env, model, max_eval, render=False, fps=60):
    # env = gym.make(env_key)
    # env = FlatObsWrapper(env)
    state = env.reset()

    model.eval()

    tot_reward = 0
    reward = 0
    n_eval = 0
    # FIXME culls out and remap actions. Find better way
    action_freq = np.zeros([7])
    while reward == 0 and n_eval < max_eval:
        state = state['image']

        # removed some scaffolding, check if something was needed
        values = model(torch.Tensor([state]))
        # values = env.step(env.action_space.sample())
        action = np.argmax(values.data.numpy()[:env.action_space.n])

        # FIXME remapping toggle action
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

    # env.close()
    if tot_reward > 0:
        print(f'action_freq: {action_freq/n_eval}\treward: {tot_reward}')
    return tot_reward


def chunks(l, n):
    ret = []
    for i in range(0, len(l), n):
        ret.append(l[i:i + n])

    return ret
