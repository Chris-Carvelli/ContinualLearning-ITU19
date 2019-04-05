import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym
import utils
import numpy as np


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MinigridRNNModule(nn.Module, torch_rl.RecurrentACModel):
    """Model is basically the same as the default torch-rl ACModel but cleaned up a bit and added method for evolution
    """

    def __init__(self, env_key):
        super().__init__()

        env = gym.make(env_key)
        action_space = env.action_space
        self.obs_space, preprocess_obss = utils.get_obss_preprocessor(env_key, env.observation_space, "")
        # self._agent = utils.Agent(env_key, env.observation_space, "", True)

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
        n = self.obs_space["image"][0]
        m = self.obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # Define memory
        self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.add_tensors = dict()
        self.init()

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(torch.tensor(obs['image']).float().unsqueeze(0), 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def evolve(self, sigma):
        named_params = self.named_parameters()
        for name, tensor in sorted(named_params):
            to_add = self.add_tensors[name]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)

    def init(self):
        # Initialize parameters correctly
        self.apply(initialize_parameters)

        named_params = self.named_parameters()
        for name, tensor in sorted(named_params):
            if name not in self.add_tensors:
                self.add_tensors[name] = torch.Tensor(tensor.size())

    def evaluate(self, env, max_eval, render=False, fps=60, show_action_frequency=False, argmax=True):
        """
        :param argmax: If True always use propabilty with highest chance as action. Other selected one at random
        :return:
        """
        state = env.reset()
        self.eval()
        memory = torch.zeros(1, self.memory_size)

        tot_reward = 0
        is_done = False
        n_eval = 0
        action_freq = np.zeros([7])
        while not is_done and n_eval < max_eval:
            dist, value, memory = self(state, memory)
            if argmax:
                action = dist.probs.max(1, keepdim=True)[1]
            else:
                action = dist.sample()

            #
            # if torch.cuda.is_available():
            #     actions = actions.cpu().numpy()

            action_freq[action] += 1
            state, reward, is_done, _ = env.step(action)

            if render:
                env.render("human")
                import time
                time.sleep(1 / fps)
            tot_reward += reward
            n_eval += 1

        if show_action_frequency:
            print(f'action_freq: {action_freq / n_eval}\treward: {tot_reward}')
        env.close()
        return tot_reward, n_eval

# if __name__ == '__main__':
#     from gym_minigrid import minigrid
#     from custom_envs import *
#
#     model = MinigridRNNModule("MiniGrid-DoorKey-5x5-v0")
#     for name, tensor in model.named_parameters():
#         print(name)
#
#     model.apply(lambda m: print(m.__class__.__name__))