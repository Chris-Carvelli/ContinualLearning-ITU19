from collections import defaultdict

import torch
import numpy as np
from torch import nn


class NTM(nn.Module):
    """An Neural Turing Machine implemention using torch for Evolution"""

    def __init__(self, memory_unit_size=4, max_memory=10000, history=False, overwrite_mode=True):
        super().__init__()
        self.nn: nn.Module = None  # The neural network. Set in inherited class
        self.min_similarity_to_jump = 0.5  # The minimum similarity required to jump to a specific location in memory
        self.shift_length = 1  # The maximum step taken when the head shifts position
        self.overwrite_mode = overwrite_mode  # If True memory writes will overwrite. Otherwise interpolate
        self.max_memory = max_memory  # The maximum length of the
        self.memory_unit_size = memory_unit_size  # The unit size of each memory cell
        self.head_pos = 0   # The position on the momory where read/write operations are currently being performed
        self.memory: np.ndarray = None  # The
        self.previous_read: np.ndarray = None
        self.left_expands: int = None  # The number of times the memory has been expanded to the left
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
        out = self.nn(x_joined).squeeze(0).detach().numpy()
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

    def evaluate(self, env, max_eval, render=False, fps=60):
        raise NotImplementedError()

    def evolve(self, sigma):
        raise NotImplementedError()
