import time

# import gym
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.modules.NTM_Module import NTM


class MinigridNTM(NTM):
    def __init__(self, memory_unit_size=None, max_memory=None):
        n = 7
        m = 7
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        super().__init__(memory_unit_size=memory_unit_size, max_memory=max_memory,
                         fixed_size=max_memory and max_memory <= 100)
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )

        self.nn = nn.Sequential(
            nn.Linear(64 + self.memory_unit_size, 64),
            nn.Tanh(),
            nn.Linear(64, 7 + self.update_size())
        )

        self.add_tensors = {}
        self.init()

    # def _nn_forward(self, x_joined):
    #     # This method is overwritten to ensure the the jump, shift and read/write parameters are normalized to [0, 1]
    #     output, update_vector = super()._nn_forward(x_joined)
    #     update_vector[:3] = 1 / (1 + np.exp(-update_vector[:3]))
    #     return output, update_vector

    def forward(self, x):
        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return super().forward(x)

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
        self.reset()
        self.eval()
        tot_reward = 0
        n_eval = 0
        is_done = False
        def convert(a):
            try:
                return tuple(convert(i) for i in a)
            except TypeError:
                return a
        # start_data = convert(state["image"]), convert(self.memory), convert(self.head_pos), convert(self.previous_read)
        past_data = set()

        while not is_done:
            # state = state['image']
            values = self(Variable(torch.Tensor([state["image"]])))
            action = np.argmax(values)

            state, reward, is_done, _ = env.step(action)
            if render:
                env.render('human')
                time.sleep(1/fps)

            tot_reward += reward
            n_eval += 1
            current_data = tuple((convert(state["image"]), convert(self.memory), self.head_pos, convert(self.previous_read)))
            no_change = past_data.__contains__(current_data)
            if no_change or n_eval >= max_eval:
                break
            past_data.add(current_data)
            # if n_eval >= max_eval: break
        return tot_reward, n_eval

if __name__ == '__main__':
    import gym
    from gym_minigrid import *
    env = gym.make("MiniGrid-Empty-5x5-v0")
    ntm = MinigridNTM(10, 10)
    total_eval = 0
    t0 = time.time()
    while total_eval < 100:
        ntm.evolve(0.05)
        # ntm.start_history()
        score, n_eval = ntm.evaluate(env, 1000000, False)
        total_eval += 1
        # print(len(ntm.memory))
        # if len(ntm.memory)> 1:
        # ntm.plot_history()
            # break
        # time.sleep(1)
    # for _ in range(10):
        # nn.

    print( (time.time() - t0) / total_eval)
