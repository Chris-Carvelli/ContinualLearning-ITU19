import os
import pickle
import sys
import time
import torch
import numpy as np

from torch.autograd import Variable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sessions.session import Session


sns.set()


def random_z_v(z_dim, z_num):
    # ret = np.random.normal(0.01, 1.0, z_dim * z_num)
    return torch.distributions.normal.Normal(torch.zeros([z_dim * z_num]), 1.0).sample()


def plot(experiment):
    s = Session(None, experiment)
    worker = s.load_results()

    gen = list(range(len(worker.med_scores)))
    s_med = worker.med_scores
    s_avg = worker.avg_scores
    s_max = worker.max_scores

    plt.plot(gen, s_med, label='med')
    plt.plot(gen, s_avg, label='avg', )
    plt.plot(gen, s_max, label='max')

    plt.legend()

    plt.show()


def simulate(env, model=None, fps=5, env_type="minigrid"):
    if env_type == "minigrid":
        state = env.reset()
        env.render()
        sys.stdout.write('Rewards:')
        while True:
            state = state['image']
            if model is not None:
                values = model(Variable(torch.Tensor([state])))
                action = np.argmax(values.data.numpy()[:env.action_space.n])
            else:
                action = env.action_space.sample()
            step = state, reward, done, info = env.step(action)
            time.sleep(1/fps)
            env.render()
            sys.stdout.write(f"{reward} ")
            if done:
                print("\nGOAL with reward: " + str(reward))
                time.sleep(1/fps)
                break
    else:
        raise NotImplementedError()
