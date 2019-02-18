import os
import pickle

import sys, os
import time
import traceback
import psutil

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


def plot(target):
    if isinstance(target, Session):
        worker = target.load_results()
    elif isinstance(target, str):
        s = Session(None, target)
        worker = s.load_results()
    else:
        worker = target

    gen = list(range(len(worker.results)))
    s_med = [r[0] for r in worker.results]
    s_avg = [r[1] for r in worker.results]
    s_max = [r[2] for r in worker.results]

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
            time.sleep(1 / fps)
            env.render()
            sys.stdout.write(f"{reward} ")
            if done:
                print("\nGOAL with reward: " + str(reward))
                time.sleep(1 / fps)
                break
    else:
        raise NotImplementedError()


def lowpriority():
    """ Set the priority of the process to below-normal. Ispired by
    https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform"""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        is_windows = False
    else:
        is_windows = True

    try:
        if is_windows:
            p = psutil.Process(os.getpid())
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(1)
    except Exception as e:
        print("Failed to save cpy priority to low. Reason:")
        traceback.print_exc()



