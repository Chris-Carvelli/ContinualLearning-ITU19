import math
import sys
import time
from pprint import pprint

import torch
from numpy.core.multiarray import ndarray
from torch import nn
from collections import defaultdict
import numpy as np


class NTM(nn.Module):
    """An Neural Turing Machine implemention using torch for Evolution"""

    def __init__(self, network: nn.Module, memory_unit_size=4, max_memory=10000, history=False, overwrite_mode=True):
        super().__init__()
        self.min_similarity_to_jump = 0.5  # The minimum similarity required to jump to a specific location in memory
        self.shift_length = 1  # The maximum step taken when the head shifts position
        self.overwrite_mode = overwrite_mode  # If True memory writes will overwrite. Otherwise interpolate
        self.max_memory = max_memory  # The maximum length of the
        self.memory_unit_size = memory_unit_size  # The unit size of each memory cell
        self.head_pos = 0   # The position on the momory where read/write operations are currently being performed
        self.memory: ndarray = None  # The
        self.previous_read: ndarray = None
        self.left_expands: int = None  # The number of times the memory has been expanded to the left
        self.network: nn.Module = network
        # History is used to plot the input, output, read-, write- operations and the location of the head.
        # History has no function impact on the model.
        # self.history: defaultdict[list] = None
        self.history: defaultdict[list] = None
        if history:
            self.history = defaultdict(list)
        self.reset()

    def reset(self):
        """Deletes all memory of the model and sets previous_read/initial_read_vector"""
        self.head_pos = 0
        self.left_expands = 0
        self.memory = np.zeros((1, self.memory_unit_size))
        self.previous_read = np.zeros(self.memory_unit_size)
        if self.history is not None:
            self.history = defaultdict(list)

    def _content_jump(self, target):
        """Shift the head position to the memory address most similar to the target vector"""
        head = self._relative_head_pos()
        similarities = 1 - np.sqrt(np.sum((self.memory - target) ** 2, 1)) / self.memory_unit_size
        pos = int(np.argmax(similarities).item())
        if similarities[pos] > self.min_similarity_to_jump:
            self.head_pos = pos
        else:
            self.head_pos = 0
        if self.history is not None:
            self.history["loc"][-1].append((head, 0.1))

    def _shift(self, s):
        """
        Shift the head position one step to the left or right a number of times determined by self.shift_length.
        If the shift causes the head position to go outside bounds of the memory it will a empty new memory unit will
        be created
        """
        start_pos = self._relative_head_pos()
        shift = int(s * 3 * self.shift_length - 0.000000001) - int(3 * self.shift_length / 2)
        for s in range(abs(shift)):
            if shift > 0:
                if self.head_pos == len(self.memory) - 1 and len(self.memory) < self.max_memory:
                    self.memory = np.concatenate((self.memory, np.zeros((1, self.memory_unit_size))), 0)
                    self.head_pos += 1
                else:
                    self.head_pos = (self.head_pos + 1) % self.max_memory
            else:
                if self.head_pos == 0 and len(self.memory) < self.max_memory:
                    self.memory = np.concatenate((np.zeros((1, self.memory_unit_size)), self.memory), 0)
                    self.left_expands += 1
                else:
                    self.head_pos = (self.head_pos - 1) % self.max_memory
        if self.history is not None:
            self.history["loc"][-1].append((start_pos, 0.1))
        return np.sign(shift)

    def _write(self, v, w):
        """
        Writes to memory at the heads position using the formula
        :param v: A vector of memory unit size
        :param w: The interpolation weight. allowed values are [0-1]. 1 completely overwrites memory
        """
        if self.overwrite_mode:
            if w > 0.5:
                self.memory[self.head_pos] = np.copy(v)
                if self.history is not None:
                    self.history["adds"][-1] = self._read()
        else:
            if self.history is not None:
                self.history["adds"][-1] = (w * (v - self._read()))
            self.memory[self.head_pos] = (1 - w) * self._read() + v * w

    def _read(self):
        """Returns the memory vector at the current head position"""
        return np.copy(self.memory[self.head_pos])

    def update_head(self, v):
        """
        Should be called once after each forward call in any implementation of NTM to update head position and load
        memory vector
        :param v: A vector of size: 3 + [memory unit size] in the format
                [shift-value, jump-value, interpolation weight] + write/jump-vector
        :return: A loaded vector from memory
        """
        s = v[0]  # shift parameter
        j = v[1]  # jump parameter
        w = v[2]  # write interpolation parameter
        m = v[3:3 + self.memory_unit_size]  # write vector
        self._write(m, w)
        if j > 0.5:
            self._content_jump(m)
        self._shift(s)
        self.previous_read = self._read()

    def update_size(self):
        """Returns the size required for the update_head(v) method to """
        return 3 + self.memory_unit_size

    def _nn_forward(self, x_joined):
        """
        This method is called by the forward method to get the outputs returned to the user and also to use as input at
        the next time step.
        Overwrite this method if you want a custom forward method the neural network of this NTM
        :param x_joined: The input received by this network. Both the read from previous timestep and the current
        user input
        :return: (output, update_vector). Output of the neural network returned to the user, and internal output used to
        update the memory at next forward call.
        """
        out = self.network(x_joined).squeeze(0).detach().numpy()
        output = out[:-self.update_size()]
        update_vector = out[-self.update_size():]
        return output, update_vector

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        if len(x.shape) > 1 and x.shape[0] == 1:
            x = x[0]
        assert len(x.shape) == 1, "Only a single sample can be forwarded at once"
        x_joined = torch.tensor(np.concatenate((x, self.previous_read), 0), dtype=torch.float64).float()
        y, v = self._nn_forward(x_joined)
        if self.history is not None:
            self.history["adds"].append(np.zeros(self.memory_unit_size))
            self.history["reads"].append(self.previous_read)
            self.history["loc"].append(list())
        self.update_head(v)
        if self.history is not None:
            self.history["in"].append(x)
            self.history["out"].append(y)
            self.history["head_pos"].append(self.head_pos)
            self.history["loc"][-1].append((self._relative_head_pos(), 1))
        return y

    def plot_history(self):
        if self.history is None:
            print("No history to plot")
            return
        import matplotlib.pyplot as plt
        n = len(self.history["head_pos"])
        m = len(self.memory)
        loc = [[0] * n for _ in range(min(self.max_memory, m + 2))]
        for i, xs in enumerate(self.history["loc"]):
            for (l, w) in xs:
                l = l % m
                loc[l][i] = min(loc[l][i] + w, 1)

        inputs = np.transpose(np.stack(self.history["in"], 0))
        outputs = np.transpose(np.stack(self.history["out"], 0))
        adds = np.transpose(np.stack(self.history["adds"], 0))
        reads = np.transpose(np.stack(self.history["reads"], 0))
        jumps = [[x for x in self.history["jumps"]]]
        shifts = [[x for x in self.history["shifts"]]]

        f, subplots = plt.subplots(3, 2, figsize=(4, 5))
        # f, subplots = plt.subplots(4, 2, figsize=(4, 8))
        subplots[0][0].imshow(inputs, vmin=0, vmax=1, cmap="gray")
        subplots[1][0].imshow(reads, vmin=0, vmax=1, cmap="gray")
        subplots[2][0].imshow(loc, vmin=0, vmax=1, cmap="hot")
        subplots[0][1].imshow(outputs, vmin=0, vmax=1, cmap="gray")
        subplots[1][1].imshow(adds, vmin=0, vmax=1, cmap="gray")
        subplots[2][1].imshow(loc, vmin=0, vmax=1, cmap="hot")
        subplots[0][0].set_ylabel('inputs')
        subplots[1][0].set_ylabel('reads')
        subplots[2][0].set_ylabel('loc')
        subplots[0][1].set_ylabel('outputs')
        subplots[1][1].set_ylabel('adds')
        subplots[2][1].set_ylabel('loc')
        # subplots[3][0].imshow(jumps, vmin=0, vmax=1, cmap="bone")
        # subplots[3][1].imshow(shifts, vmin=-1.5, vmax=1.5, cmap="twilight")
        # subplots[3][0].set_title('jumps')
        # subplots[3][1].set_title('shifts')
        for row in subplots:
            for p in row:
                p.axes.get_xaxis().set_visible(False)
                p.axes.set_yticklabels([])
                # p.axes.get_yaxis().set_visible(False)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        rect = f.patch.set_facecolor('lightsteelblue')
        plt.show()

    def _relative_head_pos(self):
        """Retunrs the head position relative to the starting position"""
        return self.head_pos - self.left_expands


class CopyNTM(NTM):
    def __init__(self, copy_size, max_memory=10, memory_unit_size=None):
        if memory_unit_size is None:
            memory_unit_size = copy_size + 2
        super().__init__(None, memory_unit_size, max_memory=max_memory)
        self.in_size = copy_size + 2
        self.out_size = copy_size
        hidden_size = 100

        self.network = nn.Sequential(
            nn.Linear(self.in_size + self.memory_unit_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, self.out_size + self.update_size()),
            nn.Sigmoid(),
        )
        self.add_tensors = {}
        self.init()

    def evolve(self, sigma):
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


class CopyNTM2(NTM):
    def __init__(self, copy_size, max_memory=10, memory_unit_size=None, update_sigma_factor=1., max_sigma=0.5):
        if memory_unit_size is None:
            memory_unit_size = copy_size + 2
        self.update_sigma_factor = update_sigma_factor
        self.max_sigma = max_sigma
        super().__init__(None, memory_unit_size, max_memory=max_memory)
        self.in_size = copy_size + 2
        self.out_size = copy_size
        self.hidden_size1 = 50
        self.hidden_size2 = 50
        self.hidden_size = self.hidden_size1 + self.hidden_size2

        self.nn_left = nn.Sequential(
            nn.Linear(self.in_size + self.memory_unit_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.out_size),
            nn.Sigmoid(),
        )

        self.nn_right = nn.Sequential(
            nn.Linear(self.in_size + self.memory_unit_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.update_size()),
            nn.Sigmoid(),
        )

        # self.nn_left1 = nn.Sequential(
        #     nn.Linear(self.in_size + self.memory_unit_size, self.hidden_size1 + self.hidden_size2),
        #     nn.Sigmoid()
        # )
        # self.nn_left2 = nn.Sequential(
        #     nn.Linear(self.hidden_size1, self.out_size),
        #     nn.Sigmoid()
        # )
        # self.nn_right1 = nn.Sequential(
        #     nn.Linear(self.in_size + self.memory_unit_size, self.hidden_size2),
        #     nn.Sigmoid()
        # )
        # self.nn_right2 = nn.Sequential(
        #     nn.Linear(self.hidden_size1 + self.hidden_size2, self.out_size),
        #     nn.Sigmoid()
        # )
        # self.nn_left1 = nn.Linear(self.in_size + self.memory_unit_size, self.hidden_size1),
        # self.nn_left2 = nn.Linear(self.hidden_size1, self.out_size),
        # self.nn_right1 = nn.Linear(self.in_size + self.memory_unit_size, self.hidden_size2),
        # self.nn_right2 = nn.Linear(self.hidden_size1 + self.hidden_size2, self.update_size()),

        self.add_tensors = {}
        self.init()

    def _nn_forward(self, x_joined):
        # out_left1 = self.nn_left1(x_joined)
        # out_right1 = self.nn_right1(x_joined)
        #
        #
        # # hidden_output = self.nn_in(x_joined)
        # # print(hidden_output)
        # output = self.nn_left2(out_left1[:self.hidden_size1]).detach().numpy()
        # update_vector = self.nn_right2(torch.cat((out_left1[self.hidden_size1:], out_right1), 0)).detach().numpy()
        output = self.nn_left(x_joined).detach().numpy()
        update_vector = self.nn_right(x_joined).detach().numpy()
        return output, update_vector

    def evolve(self, sigma):
        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            if name.startswith("nn_right"):
                to_add.normal_(0.0, min(self.update_sigma_factor * sigma, self.max_sigma))
            else:
                to_add.normal_(0.0, sigma)
            # to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                if name.startswith("nn_right"):
                    nn.init.normal_(tensor)
                else:
                    nn.init.xavier_normal_(tensor)
            else:
                tensor.data.zero_()


def evaluate_model(env, model, max_eval, render=False, fps=60, n=1, use_seed=False):
    tot_reward = 0
    if use_seed:
        if not hasattr(evaluate_model, "seed"):
            evaluate_model.seed = np.random.randint(0, 10000000)
        env.seed(evaluate_model.seed)
    for i in range(n):

        obs = env.reset()
        model.reset()
        n_eval = 0
        done = False
        while not done and n_eval < max_eval:
            y = model(obs)
            obs, reward, done, _ = env.step(y)
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
    assert ntm.head_pos == 0, f"len(ntm.head_pos) = {ntm.head_pos}"
    l = ntm.update_head(torch.tensor([1, 0, 0, 0, .5, 0, 0, 1, .8], dtype=torch.float32))  # shift & write [1, .8]*.5
    assert ntm.head_pos == 1
    assert ntm.memory[1][0] == 0.5
    assert ntm.memory[1][1] == .4


# ntm_tests()

if __name__ == '__main__':
    from custom_envs.envs import Copy

    # sys.exit()
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

    # net.evolve(0.01)
    # x = torch.randn(net.in_size).unsqueeze(0)
    # a = net(x)
    # net.evolve(0.1)
    # b = net(x)
    # print(a)
    # print(b)
    # print(a - b)
    # net = CopyNTM(12)


    rend = False
    s = np.random.randint(10000)
    net = CopyNTM2(copy_size, 12, update_sigma_factor=10, max_sigma=.5)
    for i in range(10):
        net.history = defaultdict(list)
        env.seed(s)
        evaluate_model(env, net, 1000, rend, n=1)
        net.plot_history()
        net.evolve(0.005)

    # for i in range(15):
    #     # print(net.memory)
    #     # x = torch.randint(0, 2, (1, net.in_size))
    #     x = torch.randn((1, net.in_size))
    #     # print(x)
    #     net(x)

    # pprint(dict(net.history))
    net.plot_history()
    print(net.memory)
