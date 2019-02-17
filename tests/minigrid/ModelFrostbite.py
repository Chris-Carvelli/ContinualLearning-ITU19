import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
import gym
from functools import reduce

# TODO create proper setting file (as .cfg)
Z_DIM = 32
Z_VECT_EVOLUTION_PROBABILITY = 0.5
# TODO compute in HyperNN.__init__()
Z_NUM = 4


def step(env, *args):
    state, a, b, c = env.step(*args)
    state = convert_state(state)
    return state, a, b, c


def reset(env):
    return convert_state(env.reset())


def convert_state(state):
    import cv2
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0


class HyperNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(Z_DIM, 128)
        self.l2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 4 * 4 * 64 * 512)

        self.add_tensors = {}

        self.init()

    def forward(self, x):
        x = torch.from_numpy(x).float()
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


class Model(nn.Module):
    def __init__(self, rng_state, z_v=None, hyper_mode=False):
        super().__init__()

        self.rng_state = rng_state
        self.start_z_v = z_v
        self.z_v = z_v if z_v is not None else random_z_v()
        self.hyper_mode = hyper_mode

        torch.manual_seed(rng_state)

        self.add_tensors = {}

        # hyperNN
        if hyper_mode:
            self.hyperNN = HyperNN()

        # policyNN
        self.conv1 = nn.Conv2d(4, 32, (8, 8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.dense = nn.Linear(4 * 4 * 64, 512)
        self.out = nn.Linear(512, 18)

        if hyper_mode is False:
            self.init()
        else:
            self.update_weights()

        self.evolve_states = []

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal(tensor)
            else:
                tensor.data.zero_()

    def update_weights(self):
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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, -1)
        x = F.relu(self.dense(x))

        return self.out(x)

    def evolve(self, sigma, rng_state):
        torch.manual_seed(rng_state)

        self.evolve_states.append((sigma, rng_state))

        if self.hyper_mode:
            self._evolve_hyper_mode(sigma)
        else:
            self._evolve_original(sigma)

    def _evolve_original(self, sigma):
        params = self.named_parameters()
        for name, tensor in sorted(params):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)

    def _evolve_hyper_mode(self, sigma):
        p = torch.distributions.normal.Normal(0.5, 0.1).sample().item()
        if p > Z_VECT_EVOLUTION_PROBABILITY:
            # evolve z vector
            self.z_v += torch.distributions.normal.Normal(torch.zeros([Z_DIM * Z_NUM]), sigma).sample()
        else:
            # evolve weights
            self.hyperNN.evolve(sigma)

    def compress(self):
        return CompressedModel(self.rng_state, self.start_z_v, self.evolve_states, self.hyper_mode)


def uncompress_model(model):
    start_rng, start_z_v, other_rng, hyper_mode = model.start_rng, model.start_z_v, model.other_rng, model.hyper_mode
    m = Model(start_rng, start_z_v, hyper_mode)
    for sigma, rng in other_rng:
        m.evolve(sigma, rng)

    if hyper_mode:
        m.update_weights()

    return m


def random_state():
    return random.randint(0, 2 ** 31 - 1)


def random_z_v():
    ret = np.random.normal(0.01, 1.0, Z_DIM * Z_NUM)
    return np.array(ret, dtype='double')


class CompressedModel:
    def __init__(self, start_rng=None, start_z_v=None, other_rng=None, hyper_mode=False):
        self.start_rng = start_rng if start_rng is not None else random_state()

        if hyper_mode:
            self.start_z_v = start_z_v if start_z_v is not None else random_z_v()
        else:
            self.start_z_v = None

        self.other_rng = other_rng if other_rng is not None else []
        self.hyper_mode = hyper_mode

    def evolve(self, sigma, rng_state=None):
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))


def evaluate_model(env, model, max_eval=20000, max_noop=30, render=False):
    # model = uncompress_model(model)
    noops = random.randint(0, max_noop)
    # env = gym.make(env)
    cur_states = [reset(env)] * 4
    total_reward = 0
    for _ in range(noops):
        cur_states.pop(0)
        new_state, reward, is_done, _ = step(env, 0)
        total_reward += reward
        if is_done:
            return total_reward
        cur_states.append(new_state)

    total_frames = 0
    model.eval()

    action_freq = np.zeros(env.action_space.n)

    for n_actions in range(max_eval):
        total_frames += 4
        values = model(Variable(torch.Tensor([cur_states])))[0]
        action = np.argmax(values.data.numpy()[:env.action_space.n])
        action_freq[action] += 1
        new_state, reward, is_done, _ = step(env, action)
        total_reward += reward
        if is_done:
            break
        cur_states.pop(0)
        cur_states.append(new_state)

        if render:
            env.render()

    env.close()
    return total_reward, total_frames, action_freq / (n_actions + 1)


def chunks(l, n):
    ret = []
    for i in range(0, len(l), n):
        ret.append(l[i:i + n])

    return ret
