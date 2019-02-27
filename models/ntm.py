import math
import sys
import time
from pprint import pprint

import torch
from torch import nn
from collections import defaultdict
import numpy as np


class NTM(nn.Module):
    """A Neural Turing Machine"""

    def __init__(self, network, memory_unit_size=4, max_memory=10, history=False, overwrite_mode=True):
        super().__init__()
        self.jump_threshold = 0.5
        self.min_similarity_to_jump = 0.5
        self.shift_length = 1
        self.overwrite_mode = overwrite_mode    # If True memory writes will overwrite. Otherwise interpolate
        self.max_memory = max_memory
        self.memory_unit_size = memory_unit_size
        self.head_pos = 0
        self.memory = None
        self.previous_read = None
        self.network = network
        if history:
            self.history = defaultdict(list)
        else:
            self.history = None
        self.reset()

    def reset(self):
        """Deletes all memory of the model and sets previous_read/initial_read_vector"""
        self.head_pos = 0
        self.memory = torch.zeros(self.max_memory, self.memory_unit_size)
        self.memory.requires_grad = False
        self.previous_read = torch.zeros(self.memory_unit_size)
        self.previous_read.requires_grad = False
        if self.history is not None:
            self.history = defaultdict(list)

    def _content_jump(self, target):
        """Shift the head position to the memory address most similar to the target vector"""
        # print(self.memory)
        # print(target)
        head = self.head_pos
        similarities = 1 - torch.sqrt(torch.sum((self.memory - target) ** 2, 1)) / self.memory_unit_size
        # print(self.memory - target)

        # print(similarities.data)
        pos = int(torch.argmax(similarities).item())
        if similarities[pos] > self.min_similarity_to_jump:
            self.head_pos = pos
        else:
            self.head_pos = 0
        # if head != self.head_pos:
        #     print(f"content jump from {head} to {self.head_pos}. sim = {similarities[pos]:.3f}. Target = {target}")4
        if head != self.head_pos:
            return 1
        return 0.5


    def _shift(self, s):
        """
        Shift the head position one step to the left or right a number of times determined by self.shift_length.
        If the shift causes the head position to go outside bounds of the memory it will a empty new memory unit will
        be created
        """
        start_pos = self.head_pos
        # Uniform int value from [-shift_length, shift_length]
        shift = int(s * 3 * self.shift_length - 0.000000001) - int(3 * self.shift_length / 2)
        for s in range(abs(shift)):
            if shift > 0:
                if self.head_pos == len(self.memory) - 1 and len(self.memory) < self.max_memory:
                    self.memory = torch.cat((self.memory, torch.zeros(1, self.memory_unit_size)), 0)
                    self.head_pos += 1
                else:
                    self.head_pos = (self.head_pos + 1) % self.max_memory
            else:
                if self.head_pos == 0 and len(self.memory) < self.max_memory:
                    self.memory = torch.cat((torch.zeros(1, self.memory_unit_size), self.memory), 0)
                else:
                    self.head_pos = (self.head_pos - 1) % self.max_memory
        return np.sign(shift)

    def _write(self, v, w):
        """
        Writes to memory at the heads position using the formula
        :param v: A vector of memory unit size
        :param w: The interpolation weight. allowed values are [0-1]. 1 completely overwrites memory
        """
        if self.overwrite_mode:
            if w > 0.5:
                self.memory[self.head_pos] = torch.tensor(v.detach().numpy())
                if self.history is not None:
                    self.history["adds"][-1] = self._read()
        else:
            if self.history is not None:
                self.history["adds"][-1] = torch.tensor((w *(v - self.memory[self.head_pos]).detach().numpy()))
            self.memory[self.head_pos] = torch.tensor(((1 - w) * self.memory[self.head_pos] + v * w).detach().numpy())

    def _read(self):
        """Returns the memory vector at the current head position"""
        return torch.tensor(self.memory[self.head_pos].detach().numpy())

    def update_head(self, v):
        """
        Should be called once after each forward call in any implementation of NTM to update head position and load
        memory vector
        :param v: A vector of size: 3 + [memory unit size] in the format
                [shift-value, jump-value, interpolation weight] + write/jump-vector
        :return: A loaded vector from memory
        """
        shift = v[0]
        j = v[1]
        w = v[2]
        m = v[3:3 + self.memory_unit_size]  # A memory-unit sized vector
        self._write(m, w)
        jump = 0
        if j > self.jump_threshold:
            p = self.head_pos
            jump = self._content_jump(m)
        shift = self._shift(shift)
        self.previous_read = self._read()
        if self.history is not None:
            self.history["jumps"].append(jump)
            self.history["shifts"].append(shift)


    def update_size(self):
        """Returns the size required for the update_head(v) method to """
        return 3 + self.memory_unit_size

    def forward(self, x):
        assert len(x.size()) > 1 and x.size()[0], "Only a single sample can be forwarded at once"
        x = x.double()
        x_joined = torch.cat((x.float(), self.previous_read.unsqueeze(0)), 1)
        out = self.network(x_joined).squeeze(0)
        y = out[:-self.update_size()]
        v = out[-self.update_size():]
        if self.history is not None:
            self.history["adds"].append(torch.zeros(self.memory_unit_size))
        self.update_head(v)
        if self.history is not None:
            self.history["in"].append(x.squeeze())
            self.history["out"].append(y.detach())
            self.history["head_pos"].append(self.head_pos)
            self.history["reads"].append(self.previous_read)
            # self.history["adds"].append(self._read())
        return y

    def plot_history(self):
        if self.history is None:
            print("No history to plot")
            return
        import matplotlib.pyplot as plt
        n = len(self.history["head_pos"])
        loc = [[0] * n for _ in range(self.max_memory)]
        for i, j in enumerate(self.history["head_pos"]):
            loc[j][i] = 1
        inputs = torch.transpose(torch.stack(self.history["in"], 0), 1, 0).detach()
        outputs = torch.transpose(torch.stack(self.history["out"], 0), 1, 0).detach()
        adds = torch.transpose(torch.stack(self.history["adds"], 0), 1, 0).detach()
        reads = torch.transpose(torch.stack(self.history["reads"], 0), 1, 0).detach()
        jumps = [[x for x in self.history["jumps"]]]
        shifts = [[x for x in self.history["shifts"]]]

        f, subplots = plt.subplots(4, 2, figsize=(4, 8))
        subplots[0][0].imshow(inputs, vmin=0, vmax=1, cmap="gray")
        subplots[1][0].imshow(adds, vmin=0, vmax=1, cmap="gray")
        subplots[2][0].imshow(loc, vmin=0, vmax=1, cmap="gray")
        subplots[3][0].imshow(jumps, vmin=0, vmax=1, cmap="bone")
        subplots[0][1].imshow(outputs, vmin=0, vmax=1, cmap="gray")
        subplots[1][1].imshow(reads, vmin=0, vmax=1, cmap="gray")
        subplots[2][1].imshow(loc, vmin=0, vmax=1, cmap="gray")
        subplots[3][1].imshow(shifts, vmin=-1.5, vmax=1.5, cmap="twilight")
        subplots[0][0].set_title('inputs')
        subplots[1][0].set_title('adds')
        subplots[2][0].set_title('loc')
        subplots[3][0].set_title('jumps')
        subplots[0][1].set_title('outputs')
        subplots[1][1].set_title('reads')
        subplots[2][1].set_title('loc')
        subplots[3][1].set_title('shifts')
        for row in subplots:
            for p in row:
                p.axes.get_xaxis().set_visible(False)
                p.axes.get_yaxis().set_visible(False)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        rect = f.patch.set_facecolor('lightsteelblue')
        plt.show()


class CopyNTM(NTM):
    def __init__(self, copy_size, max_memory=10, memory_unit_size=None):
        if memory_unit_size is None:
            memory_unit_size = copy_size + 2
        super().__init__(None, memory_unit_size, max_memory=max_memory)
        self.in_size = copy_size + 2
        self.out_size = copy_size
        hidden_size_1 = 100

        self.network = nn.Sequential(
            nn.Linear(self.in_size + self.memory_unit_size, hidden_size_1),
            nn.Sigmoid(),
            nn.Linear(hidden_size_1, self.out_size + self.update_size()),
            nn.Sigmoid(),
        )
        self.add_tensors = {}
        self.init()

    def get_action(self, y, env):
        """
        Transform the result of a forward(x) call to a an action for the given env
        :param y: The result of a forward operation. y=forward(x)
        :param env: The environment to perform the action
        :return: Action that can be performed the env. The dimensions are same as env.actions_space
        """
        return y

    def obs_to_input(self, obs):
        """Transform observation from copy evn to a format ready to input into this model"""
        return torch.tensor(obs, dtype=torch.float64).unsqueeze(0)

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
                nn.init.xavier_normal_(tensor)
            else:
                tensor.data.zero_()


def evaluate_model(env, model, max_eval, render=False, fps=60, n=50):
    tot_reward = 0
    for i in range(n):
        obs = env.reset()
        model.reset()
        # print(np.concatenate((env.obs[:env.length + 1] == -1, env.obs[:env.length + 1]), 0))
        # print(model.memory)
        # d = torch.tensor(list(reversed(list(env.obs[1:env.length + 1][:,2:]))))

        # model.memory[:env.length,:env.height] = d
        # print(model.memory)

        n_eval = 0
        done = False
        while not done and n_eval < max_eval:
            y = model(model.obs_to_input(obs))
            action = model.get_action(y, env)
            obs, reward, done, _ = env.step(action)
            if render:
                env.render('human')
                # print(f'action={action}, reward={reward:.2f}')
                time.sleep(1 / fps)
            tot_reward += reward
            n_eval += 1
    return tot_reward / float(n), n_eval

# TODO Fix after changes to ntm
def ntm_tests():
    """Runs unit test for the NTM class"""
    ntm = NTM(None, 2)
    ntm.history = defaultdict(list)
    ntm.memory = torch.zeros(1, ntm.memory_unit_size)
    assert len(ntm.memory) == 1
    ntm._shift(False)
    ntm._shift(True)
    ntm._shift(True)
    assert ntm.head_pos == 2
    assert len(ntm.memory) == 3
    assert torch.sum(ntm.memory) == 0
    ntm._content_jump(torch.zeros(ntm.memory_unit_size))
    assert ntm.head_pos == len(ntm.memory) - 1
    ntm.memory[1] = torch.zeros(ntm.memory_unit_size) + 1
    ntm.memory[2] = torch.zeros(ntm.memory_unit_size) + 2
    ntm._content_jump(torch.zeros(ntm.memory_unit_size))
    assert ntm.head_pos == 0
    l = ntm.update_head(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))  # Do nothing
    assert torch.sum((l - ntm.memory[ntm.head_pos])).item() == 0
    assert ntm.head_pos == 0
    l = ntm.update_head(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))  # Shift right
    assert ntm.head_pos == 1
    assert torch.sum((l - ntm.memory[ntm.head_pos])).item() == 0
    l = ntm.update_head(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))  # Shift right
    assert ntm.head_pos == 2
    l = ntm.update_head(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32))  # Content jump to [0,0]
    assert ntm.head_pos == 0
    l = ntm.update_head(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))  # shift left
    assert ntm.head_pos == 0
    assert len(ntm.memory) == 4, f"len(ntm.memory) = {len(ntm.memory)}"
    ntm._write(torch.tensor([-1.0, 1.0]), 0.2)
    assert ntm.memory[0][0] == -0.2
    assert ntm.memory[0][1] == 0.2
    ntm.head_pos = 5
    l = ntm.update_head(torch.tensor([0, 0, 0, 1, 0, -0.3, 0.1, 0, 0], dtype=torch.float32))  # Content jump to [0,0]
    assert ntm.head_pos == 0,  f"len(ntm.head_pos) = {ntm.head_pos}"
    l = ntm.update_head(torch.tensor([1, 0, 0, 0, .5, 0, 0, 1, .8], dtype=torch.float32))  # shift & write [1, .8]*.5
    assert ntm.head_pos == 1
    assert ntm.memory[1][0] == 0.5
    assert ntm.memory[1][1] == .4


# ntm_tests()

if __name__ == '__main__':
    from custom_envs.envs import Copy

    # self = NTM(None)
    # self.memory = torch.tensor([[0.4108, 0.1441, 0.2924, 0.2870, 0.3854, 0.2893],
    #     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
    # target = torch.tensor([0.4108, 0.1441, 0.2924, 0.2870, 0.3854, 0.2893])
    # self.memory_unit_size = 6
    #
    # print(self.memory)
    # print(target)
    # print(1 - (self.memory - target)**2)
    # print(torch.sum(1 - (self.memory - target) ** 2, 1) / self.memory_unit_size)
    # similarities = 1 - torch.sqrt(torch.sum((self.memory - target) ** 2, 1))/ self.memory_unit_size
    # print(similarities)
    # pos = int(torch.argmax(similarities).item())
    # print(pos)
    # print(similarities[pos])
    # # print(self.memory)
    # # print(target)
    # # head = self.head_pos
    # # similarities = torch.sqrt(torch.sum(1 - (self.memory - target) ** 2, 1)) / self.memory_unit_size
    # # print(self.memory - target)
    #
    # # print(similarities)
    # sys.exit()


    # shift = 1
    #
    # shift_length = 1
    # d = defaultdict(int)
    # for s in np.arange(0, 1.00001, 0.02):
    #     v = int(s * 3*shift_length - 0.000000001) - int(3*shift_length / 2)
    #     # v = int(np.round(s * (2.0 * shift_length) -  (2 * shift_length)/2.0))
    #     d[v] += 1
    #
    #     print(f"{s:.2f}", v)
    # pprint(dict(d))

    copy_size = 4
    env = Copy(copy_size, 6)
    net = CopyNTM(copy_size, 6)
    # x = torch.randn(net.in_size).unsqueeze(0)
    # a = net(x)
    # net.evolve(0.1)
    # b = net(x)
    # print(a)
    # print(b)
    # print(a - b)
    # net = CopyNTM(12)
    net.history = defaultdict(list)

    rend = False
    evaluate_model(env,net, 1000, rend, n=1)
    net.plot_history()
    net.evolve(0.1)
    evaluate_model(env,net, 1000, rend, n=1)

    # for i in range(15):
    #     # print(net.memory)
    #     # x = torch.randint(0, 2, (1, net.in_size))
    #     x = torch.randn((1, net.in_size))
    #     # print(x)
    #     net(x)


    # pprint(dict(net.history))
    net.plot_history()
    print(net.memory)
