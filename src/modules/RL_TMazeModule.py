import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_minigrid import minigrid
from torch.distributions.categorical import Categorical
import torch_rl
import gym


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def _unity(x):
    return x


class TMazeRLModule(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, view_size=None, recurrent=True):
        super().__init__()

        self.recurrent = recurrent


        if view_size is None:
            view_size = minigrid.AGENT_VIEW_SIZE
        if view_size <= 3:
            self.image_conv = _unity
            nn_input = view_size * view_size * 3 + 1
        else:
            nn_input = 65
            # Define image embedding
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * nn_input

        if self.recurrent:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, nn_input),
                nn.Tanh(),
                nn.Linear(nn_input, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, nn_input),
            nn.Tanh(),
            nn.Linear(nn_input, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        shape = x.shape[0]
        x = x.reshape(shape, -1)

        r = obs.reward.reshape(shape, -1)
        if x.is_cuda: r = r.cuda()
        x = torch.cat((r, x), 1)

        if self.recurrent:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory
