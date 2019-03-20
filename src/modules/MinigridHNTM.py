import numpy as np

import torch
from torch import nn

from src.modules.NTM_Module import NTM


class HyperNN(NTM):
    def __init__(self, in_size, z_num, out_size, mem_evolve_prob=0.5, mem_unit_size=4, n_fwd_pass=None, history=False):
        """

        :param in_size: z dimension
        :param z_num: size of the biggest pnn layer (not shape)
        :param out_size: max number of features
        :param mem_evolve_prob:
        :param mem_unit_size:
        """
        super().__init__(mem_unit_size, max_memory=in_size, fixed_size=True, history=history)
        # super().__init__()

        self.in_size = in_size
        self.z_num = z_num
        self.out_size = out_size
        self.mem_evolve_prob = mem_evolve_prob
        self.mem_unit_size = mem_unit_size
        self.n_fwd_pass = n_fwd_pass or in_size

        self.hidden_size = 100

        self.nn = nn.Sequential(
            nn.Linear(self.in_size + self.memory_unit_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.out_size + self.update_size()),
            nn.Tanh(),
        )
        self.add_tensors = {}
        self.init()

    def evolve(self, sigma):
        # TODO check different mem evolution strategies (all, single element, etc)
        if np.random.random() < self.mem_evolve_prob:
            self.memory += np.random.normal(np.zeros(self.memory.shape), sigma)
        else:
            for name, tensor in sorted(self.named_parameters()):
                to_add = self.add_tensors[tensor.size()]
                to_add.normal_(0.0, sigma)
                tensor.data.add_(to_add)

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                # nn.init.xavier_normal_(tensor)
                nn.init.normal_(tensor)
            else:
                tensor.data.zero_()

    def forward(self, x=None):
        out = None
        ret = []

        for i in range(self.z_num):
            for j in range(self.n_fwd_pass):
                prev_mem = self.memory[:, 0]
                out = super(HyperNN, self).forward(self.memory[:, 0])
            ret.append(out)

        return ret

