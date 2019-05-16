import random
from collections import defaultdict

import torch
from gym_minigrid import minigrid
from torch import nn

import numpy as np
from torch.autograd import Variable

from custom_envs.envs import TMaze
from src.modules.NTM_Module import NTM
from src.utils import add_min_prob, parameter_stats


class TMazeNTMModule(NTM):
    reward_inputs = 1

    def __init__(self, memory_unit_size=2, max_memory=1, reward_inputs=1, view_size: int = None,
                 layers=0, hidden_size=50):
        super().__init__(memory_unit_size, max_memory=max_memory, overwrite_mode=True)

        self.reward_inputs = reward_inputs
        if view_size is None:
            view_size = 7
        if view_size <= 3:
            self.image_conv = lambda x: x.unsqueeze(0)
            in_size = view_size * view_size * 3 + self.reward_inputs + self.memory_unit_size
        else:
            output_size = 8
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.Sigmoid(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.Sigmoid(),
                nn.Conv2d(32, output_size, (2, 2)),
                nn.Sigmoid(),
            )
            in_size = output_size + self.reward_inputs + self.memory_unit_size

        hidden_layers = []
        for _ in range(layers):
            hidden_layers.append(nn.Linear(in_size, hidden_size))
            hidden_layers.append(nn.Sigmoid())
            in_size = hidden_size
        self.nn = nn.Sequential(
            *hidden_layers,
            nn.Linear(in_size, 3 + self.update_size()),
            nn.Sigmoid(),
        )

        self.add_tensors = {}
        self.init()

    def forward(self, state):
        x = Variable(torch.Tensor([state['image']]))
        reward = state["reward"]
        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        x = torch.cat((x, torch.tensor([reward] * self.reward_inputs).float().unsqueeze(0)), 1)
        return super().forward(x)

    def evolve(self, sigma):
        named_params = list(self.named_parameters())
        evolve_vision, evolve_nn = True, True
        if any(map(lambda t: "conv" in t[0], named_params)):
            r = random.random()
            evolve_vision = 0 <= r < .333 or .666 <= r <= 1
            evolve_nn = .333 <= r <= 1
        for name, tensor in sorted(named_params):
            is_vision = name.startswith("conv")
            if (is_vision and evolve_vision) or (not is_vision and evolve_nn):
                to_add = self.add_tensors[name]
                to_add.normal_(0.0, sigma)
                tensor.data.add_(to_add)
                if ".bias" in name:
                    tensor.data.clamp_(-3, 3)
                else:
                    tensor.data.clamp_(-1, 1)

    def init(self):
        for name, tensor in self.named_parameters():
            if name not in self.add_tensors:
                self.add_tensors[name] = torch.Tensor(tensor.size())
            if 'weight' in name:
                tensor.data.zero_()
            else:
                nn.init.normal_(tensor, std=.5)
                # nn.init.kaiming_normal_(tensor)
                #                 # if name.startswith("conv"):
                #                 #     nn.init.xavier_normal(tensor)
                #                 #     # nn.init.kaiming_normal_(tensor)
                #                 # else:
                #                 #     nn.init.normal_(tensor)

    def evaluate(self, env, max_eval, render=False, fps=60, show_action_frequency=False, random_actions=False,
                 mode="human"):
        state = env.reset()
        self.reset()
        self.eval()

        tot_reward = 0
        is_done = False
        n_eval = 0
        action_freq = np.zeros([7])
        prop_product = 1
        while not is_done and n_eval < max_eval:
            values = self(state)
            if random_actions:
                p = add_min_prob(values, 0.0)
                action = np.random.choice(np.array(range(len(p)), dtype=np.int), p=p)
                prop_product *= p[action]
            else:
                action = np.argmax(values)
            action_freq[action] += 1
            state, reward, is_done, _ = env.step(action)

            if render:
                env.render(mode)
                import time
                time.sleep(1 / fps)
            if reward > 0:
                tot_reward += reward * prop_product
                prop_product = 1
            # tot_reward += reward
            n_eval += 1

        if show_action_frequency:
            print(f'action_freq: {action_freq / n_eval}\treward: {tot_reward}')
        env.close()
        return tot_reward, n_eval


if __name__ == '__main__':

    from custom_envs import *
    import gym
    import time

    # env = gym.make("TMaze-1x2x12-v0")
    env: TMaze = gym.make("TMaze-3x2-viewsize_3-v0")

    ntm = TMazeNTMModule(memory_unit_size=2, max_memory=3, view_size=3, layers=1, hidden_size=50)
    print(ntm)

    # ntm.divergence = 1

    params = torch.nn.utils.parameters_to_vector(ntm.parameters()).detach().numpy()
    print(len(params))
    ntm.evolve(.1)

    while True:

        ntm.start_history()
        r, n_eval = ntm.evaluate(env, 1000, render=False, fps=30, show_action_frequency=False)
        # ntm.plot_history(vmin=0, vmax=1)
        if r > 0:
            print(r, n_eval),
            ntm.evaluate(env, 1000, render=True, fps=10, show_action_frequency=True)
        # ntm.evaluate(env, 1000, render=False, fps=10)
        ntm.evolve(.1)
        # parameter_stats(ntm, False)

    # while ntm.evaluate(env, 30)[0] <= 0.5:
    #     ntm.history = defaultdict(list)
    #     ntm.evaluate(env, 1000, False, fps=12)
    #     ntm.evolve(.5)
    #     # print("round")
    #     ntm.plot_history()
    # ntm.history = defaultdict(list)
    # print(ntm.evaluate(env, 1000))
    # ntm.plot_history()
    # while True:
    #     ntm.history = None
    #     print(ntm.evaluate(env, 1000, True, fps=3))
    #     time.sleep(.5)

    # Test dill serialization
    # import dill
    # import sys
    # from pathlib import Path
    # dill.dump(ntm, open(Path(sys.argv[0]).parent / "test.dill", "wb"))
    # ntm2 = dill.load(open(Path(sys.argv[0]).parent / "test.dill", "rb"))
    # print(ntm)
    # print(ntm2)
