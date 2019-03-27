import dill
import gym
from custom_envs import *
from src.ga import GA


from src.Controllers.ControllerHyper import Controller

from src.modules.MinigridPolicy import PolicyNN
from src.modules.MinigridHNTM import HyperNN

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print('main')

    ga = GA(
            env_key='TMaze-2x10-v0',
            model_builder=lambda: Controller(
                PolicyNN(),
                HyperNN(
                    in_size=32,
                    z_num=4,
                    out_size=32 * 64 * 2 * 2,
                    mem_evolve_prob=0.5,
                    n_fwd_pass=32
                ),
            ),
            population=500,
            max_generations=20,
            max_episode_eval=10000,
            sigma=0.005,
            truncation=7,
            elite_trials=5,
            n_elites=1)

    res = True
    while res is not False:
        res = ga.iterate()

        res[-1][0][0].evaluate(ga.env_key, 1000, True)
        dill.dump(ga, open('ga.dill', 'w+b'))
        dill.dump(res, open('res.dill', 'a+b'))
        print(res)


def range_evol(n):
    from collections import defaultdict

    z_dim = 32
    z_num = 4
    out_features = 32 * 64 * 2 * 2
    hntm = HyperNN(z_dim, z_num, out_features, mem_evolve_prob=0.5, n_fwd_pass=32)
    for i in range(n):
        hntm.history = defaultdict(list)
        hntm.evolve(0.1)

    return hntm


def plot_w(r, shape):
    f, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat):
        sns.heatmap(r[i].reshape(shape), ax=ax, square=True)
    plt.show()


if __name__ == "__main__":
    env = gym.make('TMaze-2x10-v0')
    c = Controller(
                PolicyNN(),
                HyperNN(
                    in_size=32,
                    z_num=4,
                    out_size=32 * 64 * 2 * 2,
                    mem_evolve_prob=0.5,
                    n_fwd_pass=32
                ),
            )

    c.evaluate(env, 10000, True)
    # main()
