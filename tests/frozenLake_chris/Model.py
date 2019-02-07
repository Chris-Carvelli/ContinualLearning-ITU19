import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gym


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.l1 = nn.Linear(500, 128)
        self.l2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 6)

    def forward(self, x):
        # x = torch.from_numpy(x).float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

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
                nn.init.kaiming_normal(tensor)
            else:
                tensor.data.zero_()


def evaluate_model(env_key, model):
    env = gym.make(env_key)
    state = env.decode(env.reset())

    model.eval()

    tot_reward = 0
    done = False
    while not done:
        # removed some scaffolding, check if something was needed
        values = model(Variable(torch.Tensor([state])))
        # values = env.step(env.action_space.sample())
        action = np.argmax(values.data.numpy()[:env.action_space.n])

        state, reward, is_done, _ = env.step(env.encode(action))
        tot_reward += reward

    env.close()
    return tot_reward
