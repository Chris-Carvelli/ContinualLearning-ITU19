import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import random_z_v


class HyperNN(nn.Module):
    def __init__(self, z_dim, z_num, out_features, z_v_evolve_prob):
        super().__init__()

        self.z_dim = z_dim
        self.z_num = z_num
        self.z_v_evolve_prob = z_v_evolve_prob

        self.nn = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )
        self.z_v = random_z_v(self.z_dim, self.z_num)

        self.add_tensors = {}

        self.init()

    def forward(self, layer_index):
        if layer_index is None:
            return [self.nn(x) for x in self.z_v]
        else:
            return self.nn(self.z_v[layer_index])

    def evolve(self, sigma):
        p = torch.distributions.normal.Normal(0.5, 0.1).sample().item()
        if p > self.z_v_evolve_prob:
            # evolve z vector
            self.z_v += torch.distributions.normal.Normal(
                torch.zeros([self.z_dim * self.z_num]),
                sigma
            ).sample()
        else:
            # evolve weights
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
                nn.init.kaiming_normal_(tensor)
            else:
                tensor.data.zero_()
