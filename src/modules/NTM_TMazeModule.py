import torch
from torch import nn

import numpy as np
from torch.autograd import Variable

from src.modules.NTM_Module import NTM


class TMazeNTMModule(NTM):
    def __init__(self, memory_unit_size, max_memory=None,):
        super().__init__(memory_unit_size, max_memory=max_memory)

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.Sigmoid(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.Sigmoid(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Sigmoid(),
        )

        hidden_size = 100
        self.nn = nn.Sequential(
            nn.Linear(64 + 1 + self.memory_unit_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 4 + self.update_size()),
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

        x = torch.cat((x, torch.tensor([reward]).float().unsqueeze(0)), 1)
        return super().forward(x)


    def evolve(self, sigma):
        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[name]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)

    def init(self):
        for name, tensor in self.named_parameters():
            if name not in self.add_tensors:
                self.add_tensors[name] = torch.Tensor(tensor.size())
            if 'weight' in name:
                tensor.data.zero_()
            elif name.startswith("conv"):
                nn.init.xavier_normal(tensor)
                # nn.init.kaiming_normal_(tensor)
            else:
                nn.init.normal_(tensor)

    def evaluate(self, env, max_eval, render=False, fps=60):
        # env = gym.make(env_key)
        # env = FlatObsWrapper(env)
        state = env.reset()
        self.reset()
        self.eval()

        tot_reward = 0
        is_done = False
        n_eval = 0
        action_freq = np.zeros([7])
        while not is_done and n_eval < max_eval:
            values = self(state)
            action = np.argmax(values)
            if action is 3:
                action = 5

            action_freq[action] += 1
            state, reward, is_done, _ = env.step(action)

            if render:
                env.render('human')
                # print('action=%s, reward=%.2f' % (action, reward))
                import time
                time.sleep(1/fps)

            tot_reward += reward
            n_eval += 1

        # env.close()
        # if tot_reward > 0:
            # print(f'action_freq: {action_freq/n_eval}\treward: {tot_reward}')
        return tot_reward, n_eval

if __name__ == '__main__':

    from custom_envs import *
    import gym

    env = gym.make("TMaze-2x5-v0")

    ntm = TMazeNTMModule(6)
    ntm.init()

    while ntm.evaluate(env, 30)[0] == 0:
        ntm.evolve(.1)
        print("round")
    while True:
        ntm.evaluate(env, 30, True, fps=5)
        import time
        time.sleep(2)

    # Test dill serialization
    # import dill
    # import sys
    # from pathlib import Path
    # dill.dump(ntm, open(Path(sys.argv[0]).parent / "test.dill", "wb"))
    # ntm2 = dill.load(open(Path(sys.argv[0]).parent / "test.dill", "rb"))
    # print(ntm)
    # print(ntm2)


