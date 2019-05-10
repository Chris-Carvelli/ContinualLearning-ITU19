import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class PolicyNN(nn.Module):
    def __init__(self, obs_space, action_space):
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        super(PolicyNN, self).__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        self.add_tensors = {}
        self.init()

    def forward(self, x):
        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.actor(x)

        return x

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

    def evaluate(self, env, max_eval, render=False, fps=60):
        state = env.reset()

        self.eval()

        tot_reward = 0
        n_eval = 0
        is_done = False

        while not is_done:
            state = state['image']

            # removed some scaffolding, check if something was needed
            values = self(Variable(torch.Tensor([state])))
            action = np.argmax(values.data.numpy())

            state, reward, is_done, _ = env.step(action)
            if render:
                # print(f'render eval {n_eval}')
                # print('action=%s, reward=%.2f' % (action, reward))
                env.render('human')
                time.sleep(1/fps)

            tot_reward += reward
            n_eval += 1
            if max_eval > -1 and n_eval >= max_eval:
                break

        return tot_reward, n_eval
