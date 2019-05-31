from functools import reduce
import torch
import torch.nn as nn

from src.modules.PolicyNN import PolicyNN


def random_z_v(z_dim, z_num):
    # ret = np.random.normal(0.01, 1.0, z_dim * z_num)
    return torch.distributions.normal.Normal(torch.zeros([z_num, z_dim]), 0.1).sample()


class HyperNN(nn.Module):
    def __init__(self):
        super().__init__()

        self._tiling = False

        self.z_dim = 32
        self.z_v_evolve_prob = 0.5

        self.pnn = PolicyNN()
        self.out_features = self._get_out_features()
        self.z_num = self._get_z_num()

        self.nn = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_features),
        )
        self.z_v = random_z_v(self.z_dim, self.z_num)

        self.add_tensors = {}

        self.init()

    def forward(self, layer_index=None):
        if layer_index is None:
            return [self.nn(x) for x in self.z_v]
        else:
            return self.nn(self.z_v[layer_index])

    def evolve(self, sigma):
        p = torch.distributions.normal.Normal(0.5, 0.1).sample().item()
        if p > self.z_v_evolve_prob:
            # evolve z vector
            self.z_v += torch.distributions.normal.Normal(
                torch.zeros([self.z_num,self.z_dim]),
                sigma
            ).sample()
        else:
            # evolve weights
            params = self.named_parameters()
            for name, tensor in sorted(params):
                to_add = self.add_tensors[tensor.size()]
                to_add.normal_(0.0, sigma)
                tensor.data.add_(to_add)

        self._update_weights()

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal_(tensor)
            else:
                tensor.data.zero_()

        self._update_weights()

    def evaluate(self, env, max_eval, render=False, fps=60):
        return self.pnn.evaluate(env, max_eval, render, fps)

    # tiling not supported (but it should be a bit faster, performance gain unclear)
    def _update_weights(self):
        weights = self()
        i = 0
        for name, param in self.pnn.named_parameters():
            if 'weight' in name:
                param.data = self._shape_w(weights[i], param.shape).data
                i += 1

    def _shape_w(self, w, layer_shape):
        w = torch.Tensor(w)
        w = torch.narrow(w, 0, 0, reduce((lambda x, y: x * y), layer_shape))
        w = w.view(layer_shape)

        return w

    def _get_z_num(self):
        z_num = 0

        # tiling
        for name, param in self.pnn.named_parameters():
            if 'weight' in name:
                z_num += 1
                # tiling
                # layer_shape = param.shape
                # layer_size = reduce((lambda x, y: x * y), layer_shape)
                # z_num += layer_size // self.out_features + 1

        return z_num

    def _get_out_features(self):
        if self._tiling:
            return 64

        ret = 0
        for name, param in self.pnn.named_parameters():
            if 'weight' in name:
                layer_shape = param.shape
                layer_size = reduce((lambda x, y: x * y), layer_shape)
                if layer_size > ret:
                    ret = layer_size
        return ret

