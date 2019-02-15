import os
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()


def random_z_v(z_dim, z_num):
    # ret = np.random.normal(0.01, 1.0, z_dim * z_num)
    return torch.distributions.normal.Normal(torch.zeros([z_dim * z_num]), 1.0).sample()


def plot(env, experiment):
    path = os.path.join(
        os.getcwd(),
        # f'Experiments/{env}/{experiment}/process.pickle'
        'process.pkl'
    )

    fp = open(path, 'rb')
    data = []

    while True:
        try:
            data.append(pickle.load(fp))
        except EOFError:
            fp.close()
            break

    gen = list(range(len(data)))
    s_med = [d[0] for d in data]
    s_avg = [d[1] for d in data]
    s_max = [d[2] for d in data]

    plt.plot(gen, s_med, label='med')
    plt.plot(gen, s_avg, label='avg', )
    plt.plot(gen, s_max, label='max')

    plt.legend()

    plt.show()
