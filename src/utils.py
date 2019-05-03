import configparser
import os
import traceback
from typing import List
import numpy as np
import dill
import importlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


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
    return 0.5 + np.sign(u) * (redistribute(np.abs(u) * 2, f) / 2)


def add_min_prob(w: np.ndarray, min_prob=0):
    """
    Returns a probability distribution p based on w
    :param w: Weights
    :param min_prob: The minimum probability allowed in the probability distribution. if min_prop=0 the returned values
    will simply be  w / sum(w)
    :return: probability distribution
    """
    assert min_prob < 1 / len(w)
    w = (w + np.sum(w) * min_prob / (1 - len(w) * min_prob))
    return w / np.sum(w)


def parameter_stats(nn: nn.Module, print_indivual_params=True):
    """Prints various statistics about the paramters of a neural neural network"""
    p = torch.nn.utils.parameters_to_vector(nn.parameters()).detach().numpy()
    print(f"max/min = {max(p):.1f}/{min(p):.1f}, mean={np.mean(p):.1f}+/-{np.std(p):.2f}")
    if print_indivual_params:
        for name, p in nn.named_parameters():
            p = p.detach().numpy()
            print(f"{name:40s}max/min = {np.max(p):.1f}/{np.min(p):.1f}, mean={np.mean(p):.1f}+/-{np.std(p):.2f}")


def model_diff(models: List[nn.Module], models2: List[nn.Module] = None, verbose=True):
    """Prints out the standard deviation between parameters of the supplied models in the form of a NxN matrix"""
    if models2 is not None:
        n = len(models)
        assert len(models) == len(models2)
        std_array = np.zeros(n)
        for i in range(n):
            params1 = torch.nn.utils.parameters_to_vector(models[i].parameters()).detach().numpy()
            params2 = torch.nn.utils.parameters_to_vector(models2[i].parameters()).detach().numpy()
            std_array[i] = np.std(params1 - params2)
    else:
        std_array = np.array([[0 for _ in range(len(models))] for _ in range(len(models))], dtype=np.float)
        for i, m1 in enumerate(models):
            params1 = torch.nn.utils.parameters_to_vector(m1.parameters()).detach().numpy()
            for j, m2 in enumerate(models[:i]):
                params2 = torch.nn.utils.parameters_to_vector(m2.parameters()).detach().numpy()
                std = np.std(params1 - params2)
                std_array[i, j] = 0
                std_array[j, i] = std
    if verbose:
        print(np.round(std_array, 2))
    return std_array


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
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(1)
    except Exception as e:
        print("Failed to save cpy priority to low. Reason:")
        traceback.print_exc()


def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


def split_permutations(n, minimum=0, recurse=0, require_full_size=False):
    """
    Generates a list of permutations for how many ways you can split a the number n into smaller values
    For example:
        split_permutations(5, minimum=2, recurse=0) = [[2, 3], [3, 2]]
        split_permutations(4, minimum=1, recurse=1) = [[1, 2, 1], [1, 3], [2, 1, 1], [3, 1], [1, 1, 1, 1], [1, 1, 2], [2, 2]]
        split_permutations(4, minimum=1, recurse=1, require_full_size=True) = [[1, 1, 1, 1]]
    :param n: The number to split
    :param minimum: The minimum allowed value for a split
    :param recurse: Also split sub results
    :param require_full_size: If true only allow full permutations that are fully split(see example above)
    :return: list split of permutations
    """
    if n < minimum * 2:
        return []
    if recurse <= 0:
        return [[i, n - i] for i in range(minimum, n - minimum + 1)]
    else:
        p = set()
        for i in range(minimum, n // 2 + 1):
            s1 = split_permutations(i, minimum, recurse - 1)
            s2 = split_permutations(n - i, minimum, recurse - 1)
            if not require_full_size:
                s1.append([i])
                s2.append([n - i])
            for a in s1:
                for b in s2:
                    p.add(tuple(a + b))
                    p.add(tuple(b + a))
        return [list(x) for x in p]
