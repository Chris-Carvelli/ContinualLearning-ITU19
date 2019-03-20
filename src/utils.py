import dill
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()


def random_z_v(z_dim, z_num):
    # ret = np.random.normal(0.01, 1.0, z_dim * z_num)
    return torch.distributions.normal.Normal(torch.zeros([z_num, z_dim]), 0.1).sample()


def plot(target):
    runs = ['hyper', 'base']
    cols = ['run', 'exp', 'gen', 'med', 'avg', 'max', 'frames']
    data = []
    for run in runs:
        for exp in range(5):
            fp = open(f'{target}/{run}/run_{exp}.pkl', 'rb')

            ga = dill.load(fp)

            for g, res in enumerate(ga.results):
                # TMP
                data.append([run, exp, g, res[0], res[1], res[2], res[3]])

    df = pd.DataFrame(data, columns=cols)
    sns.lineplot(x='gen', y='max', hue='run', data=df,
                 err_style="bars", ci='sd')
    plt.legend()

    plt.show()


def chunks(l, n):
    ret = []
    for i in range(0, len(l), n):
        ret.append(l[i:i + n])

    return ret


def redistribute(v: np.ndarray, f=1):
    """Accepts an input vector v of values in range [0, 1]. The sharpening factor f determines how much the
    values are pushed towards 1 (f=1 means no change)"""
    with np.errstate(divide='ignore'):
        return 1 / (1 + (1 / v - 1)) ** (1 / f)


def diverge(v: np.ndarray, f=1):
    """Accepts and input vector v with values in range [0, 1]. Values above 0.5 are pushed towards 1 and value below
    0.5 will be pushed towards 0. f determines how much values are pushed towards """
    u = v - 0.5
    return 0.5 + np.sign(u) * (redistribute(np.abs(u)*2, f) / 2)


def add_min_prob(w: np.ndarray, min_prob=0):
    """
    Returns a probability distribution p based on w
    :param w: Weights
    :param min_prob: The minimum probability allowed in the probability distribution. if min_prop=0 the returned values
    will simply be  w / sum(w)
    :return: probability distribution
    """
    assert min_prob < 1/len(w)
    w = (w + np.sum(w) * min_prob / (1 - len(w) * min_prob))
    return w / np.sum(w)

