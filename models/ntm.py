import time
from pprint import pprint

import torch
from torch import nn
from collections import defaultdict


class NTM(nn.Module):
    """A Neural Turing Machine"""
    verbose = False

    def __init__(self, network, memory_unit_size=4, max_memory=10, initial_read_vector=None, history=False):
        super(NTM, self).__init__()
        self.jump_threshold = 0.5
        self.max_memory = max_memory
        self.memory_unit_size = memory_unit_size
        self.head_pos = 0
        self.memory = torch.zeros(self.max_memory, self.memory_unit_size)
        self.previous_read = initial_read_vector
        if initial_read_vector is None:
            self.previous_read = torch.zeros(self.memory_unit_size)
        self.network = network
        if history:
            self.history = defaultdict(list)
        else:
            self.history = None


    def update_size(self):
        return 2 * self.memory_unit_size + 5

    def content_jump(self, target):
        """Shift the head position to the memory address most similar to the target vector"""
        if self.verbose: print(f"content jump from {self.head_pos} to "
                               f"{int(torch.argmin(torch.sqrt(torch.sum((self.memory - target) ** 2, 1))).item())}")
        self.head_pos = int(torch.argmin(torch.sqrt(torch.sum((self.memory - target) ** 2, 1))).item())

    def shift(self, right=True):
        """
        Shift the head position one step to the left or right.
        If the shift causes the head position to go outside bounds of the memory it will a empty new memory unit will
        be created
        :param right: If False the shift will be towards the left
        """
        start_pos = self.head_pos
        if right:
            if self.verbose: print(f"shift right from {self.head_pos}")
            if self.head_pos == len(self.memory) - 1 and len(self.memory) < self.max_memory:
                self.memory = torch.cat((self.memory, torch.zeros(1, self.memory_unit_size)), 0)
                self.head_pos += 1
            else:
                self.head_pos = (self.head_pos + 1) % self.max_memory
        else:
            if self.verbose: print(f"shift left from {self.head_pos}")
            if self.head_pos == 0 and len(self.memory) < self.max_memory:
                self.memory = torch.cat((torch.zeros(1, self.memory_unit_size), self.memory), 0)
            else:
                self.head_pos = (self.head_pos - 1) % self.max_memory
        return start_pos - self.head_pos

    def write(self, v, w):
        """
        Writes to memory at the heads position using the formula
        :param v: A vector of memory unit size
        :param w: The interpolation weight. allowed values are [0-1]. 1 completely overwrites memory
        """
        if self.verbose: print(f"Wrote {v} to position {self.head_pos} with w={w}")
        self.memory[self.head_pos] = (1 - w) * self.memory[self.head_pos].clone() + v * w

    def read(self):
        """Returns the memory vector at the current head position"""
        return self.memory[self.head_pos].clone()

    def update_head(self, v):
        """
        Should be called once after each forward call in any implementation of NTM to update head position and load
        memory vector
        :param v: A vector of size: 5 + 2*[memory unit size] in the format
                [shift-right, shift_left, shift-stay, content-jump] + [jump target] + [
        :return: A loaded vector from memory
        """
        shift = torch.argmax(v[0:2])
        jump = v[3]
        w = v[4]
        if jump > self.jump_threshold:
            self.content_jump(v[5:5 + self.memory_unit_size])
        if 0 <= shift < 2:
            self.shift((int(shift) == 0))
        ret = self.read()
        self.write(v[5 + self.memory_unit_size:], w)
        return ret

    def forward(self, x):
        assert len(x.size()) > 1 and x.size()[0], "Only a single sample can be forwarded at once"
        x = x.double()
        # print(x, self.previous_read.unsqueeze(0).double())
        x_joined = torch.cat((x.float(), self.previous_read.unsqueeze(0)), 1)
        out = self.network(x_joined).squeeze(0)
        y = out[:-self.update_size()]
        v = out[-self.update_size():]
        self.previous_read = self.update_head(v)
        if self.history is not None:
            self.history["in"].append(x.squeeze())
            self.history["out"].append(y)
            self.history["head_pos"].append(self.head_pos)
            self.history["reads"].append(self.previous_read)
            self.history["adds"].append(self.memory[self.head_pos].clone() - self.previous_read)
        return y

    def plot_history(self):
        if self.history is None:
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

        f, subplots = plt.subplots(3, 2, figsize=(4, 8))
        subplots[0][0].imshow(inputs, vmin=0, vmax=1)
        subplots[1][0].imshow(adds, vmin=0, vmax=1)
        subplots[2][0].imshow(loc, vmin=0, vmax=1)
        subplots[0][1].imshow(outputs, vmin=0, vmax=1)
        subplots[1][1].imshow(reads, vmin=0, vmax=1)
        subplots[2][1].imshow(loc, vmin=0, vmax=1)
        subplots[0][0].set_title('inputs')
        subplots[1][0].set_title('adds')
        subplots[2][0].set_title('loc')
        subplots[0][1].set_title('outputs')
        subplots[1][1].set_title('reads')
        subplots[2][1].set_title('loc')
        for row in subplots:
            for p in row:
                p.axes.get_xaxis().set_visible(False)
                p.axes.get_yaxis().set_visible(False)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.show()


class CopyNTM(NTM):
    def __init__(self, copy_size, max_memory=10):
        super().__init__(None, copy_size + 2, max_memory=max_memory)
        self.in_size = copy_size + 2
        self.out_size = copy_size
        hidden_size_1 = self.out_size * 2
        hidden_size_2 = self.out_size * 2

        self.network = nn.Sequential(
            nn.Linear(self.in_size + self.memory_unit_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, self.out_size + self.update_size()),
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
                nn.init.kaiming_normal_(tensor)
            else:
                tensor.data.zero_()


def evaluate_model(env, model, max_eval, render=False, fps=60):
    obs = env.reset()
    n_eval = 0
    tot_reward = 0
    done = False
    while not done and n_eval < max_eval:
        y = model(model.obs_to_input(obs))
        action = model.get_action(y, env)
        obs, reward, done, _ = env.step(action)
        if render:
            env.render('human')
            print(f'action={action}, reward={reward:.2f}')
            time.sleep(1/fps)
        tot_reward += reward
        n_eval += 1
    return tot_reward, n_eval


def ntm_tests():
    """Runs unit test for the NTM class"""
    # TODO: Fix tests. Does not work after chages to ntm.update_head
    ntm = NTM(2)
    assert len(ntm.memory) == 1
    ntm.shift(False)
    ntm.shift(True)
    ntm.shift(True)
    assert ntm.head_pos == 2
    assert len(ntm.memory) == 3
    assert torch.sum(ntm.memory) == 0
    ntm.content_jump(torch.zeros(ntm.memory_unit_size))
    assert ntm.head_pos == len(ntm.memory) - 1
    ntm.memory[1] = torch.zeros(ntm.memory_unit_size) + 1
    ntm.memory[2] = torch.zeros(ntm.memory_unit_size) + 2
    ntm.content_jump(torch.zeros(ntm.memory_unit_size))
    assert ntm.head_pos == 0
    l = ntm.update_head(torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32))  # Do nothing
    assert torch.sum((l - ntm.memory[ntm.head_pos])).item() == 0
    assert ntm.head_pos == 0
    l = ntm.update_head(torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32))  # Shift right
    assert ntm.head_pos == 1
    assert torch.sum((l - ntm.memory[ntm.head_pos])).item() == 0
    l = ntm.update_head(torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32))  # Shift right
    assert ntm.head_pos == 2
    l = ntm.update_head(torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32))  # Content jump to [0,0]
    assert ntm.head_pos == 0
    l = ntm.update_head(torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32))  # shift left
    assert ntm.head_pos == 0
    assert len(ntm.memory) == 5
    ntm.write(torch.tensor([-1.0, 1.0]), 0.2)
    assert ntm.memory[0][0] == -0.2
    assert ntm.memory[0][1] == 0.2
    ntm.head_pos = 5
    l = ntm.update_head(torch.tensor([0, 0, 0, 1, -0.3, 0.1], dtype=torch.float32))  # Content jump to [0,0]
    assert ntm.head_pos == 0


if __name__ == '__main__':
    net = CopyNTM(8)
    net.history = defaultdict(list)
    for i in range(10):
        net.forward(torch.rand(net.in_size).unsqueeze(0))

    pprint(dict(net.history))
    net.plot_history()
