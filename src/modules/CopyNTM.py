import torch
from torch import nn

from src.modules.NTM_Module import NTM


class CopyNTM(NTM):
    clip: float = None

    def __init__(self, copy_size=None, max_memory=10000, memory_unit_size=None, clip=None, layers=1, hidden_size=100):
        if memory_unit_size is None:
            memory_unit_size = copy_size + 2
        super().__init__(memory_unit_size, max_memory=max_memory)
        self.clip = clip
        self.in_size = copy_size + 2
        self.out_size = copy_size
        assert layers >= 1

        args = [nn.Linear(self.in_size + self.memory_unit_size, hidden_size), nn.Sigmoid()]
        for _ in range(layers - 1):
            args += [nn.Linear(hidden_size, hidden_size), nn.Sigmoid()]
        args += [nn.Linear(hidden_size, self.out_size + self.update_size()), nn.Sigmoid()]
        self.nn = nn.Sequential(*args)
        self.add_tensors = {}
        self.init()

    def evolve(self, sigma):
        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)
            if self.clip:
                if ".bias" in name:
                    tensor.data.clamp_(-3*self.clip, 3*self.clip)
                else:
                    tensor.data.clamp_(-self.clip, self.clip)

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)
                # nn.init.normal_(tensor, std=self.init_sigma)
            else:
                tensor.data.zero_()

    def evaluate(self, env, max_eval, render=False, fps=60):
        tot_reward = 0
        obs = env.reset()
        self.reset()
        n_eval = 0
        done = False
        while not done and n_eval < max_eval:
            y = self(obs)
            obs, reward, done, _ = env.step(y)
            if render:
                env.render('human')
            tot_reward += reward
            n_eval += 1
        return tot_reward, n_eval
