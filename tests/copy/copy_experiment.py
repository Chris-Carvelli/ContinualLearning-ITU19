from collections import defaultdict

import psutil, os
import gym

from custom_envs import *
from models.ntm import CopyNTM, evaluate_model
from sessions.session import Session
from tests.minigrid.ga import GA
from tests.minigrid.utils import *

copy_size = 2
length = 4


class MyModel(CopyNTM):
    def __init__(self, hypermodule=False):
        super().__init__(copy_size, 2 * length + 2)


def main():
    # Sets CPU usage priority to low

    lowpriority()
    env_key = f"Copy-{copy_size}x{length}-v0"

    ga = GA(env_key, 500, max_generations=300,
            sigma=0.010,
            truncation=10,
            elite_trials=5,
            n_elites=5)
    ga.Model = MyModel
    ga.evaluate_model = evaluate_model

    # TODO: Find better name (my PC trims the last past of the name away)
    name = f"{env_key}_05_{ga.population}_{ga.sigma}_{ga.n_elites}"
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
