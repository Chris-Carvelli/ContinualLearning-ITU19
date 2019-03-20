import dill

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
