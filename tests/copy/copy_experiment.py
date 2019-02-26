from collections import defaultdict

import psutil, os
import gym

from custom_envs import *
from models.ntm import CopyNTM, evaluate_model
from sessions.session import Session
from tests.minigrid.ga import GA
from tests.minigrid.utils import *

copy_size = 1
length = 12


class MyModel(CopyNTM):
    def __init__(self, hypermodule=False):
        super().__init__(copy_size, 2 * length + 2, memory_unit_size=copy_size * 2 + 2)


def main():
    # Sets CPU usage priority to low

    lowpriority()
    # env_key = f"Copy-{copy_size}x{length}-v0"
    env_key = f"CopyRnd-{copy_size}-v0"

    ga = GA(env_key, 300, max_generations=5000,
            sigma=0.05,
            truncation=10,
            elite_trials=5,
            n_elites=10)
    ga.Model = MyModel
    ga.evaluate_model = evaluate_model

    name = f"{env_key}_10_{ga.population}_{ga.sigma}_{ga.n_elites}"
    session = Session(ga, name)

    session.start()

    ga = session.load_results()
    plot(ga)
    champ = ga.results[-1][-1][0][0]
    for x in range(5):
        champ.history = defaultdict(list)
        res = evaluate_model(ga.env, champ, 100000, n=1)
        print(res)
        champ.plot_history()

    # env = gym.make(ga.env_key)
    # parents = [p for p, _ in filter(lambda x: x[1] > 0, ga.scored_parents)]
    # model = parents[0]
    # while True:
    #     simulate(env, model, fps=4)


if __name__ == "__main__":
    main()
