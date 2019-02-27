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
        super().__init__(copy_size, 2 * length + 2, memory_unit_size=copy_size * 2)


def main():
    # Sets CPU usage priority to low
    lowpriority()

    # env_key = f"Copy-{copy_size}x{length}-v0"
    env_key = f"CopyRnd-{copy_size}-v0"

    ga = GA(env_key, 500, max_generations=5000,
            sigma=0.005,
            truncation=10,
            elite_trials=5,
            n_elites=10)
    ga.Model = MyModel
    ga.evaluate_model = evaluate_model

    name = f"{env_key}_01_{ga.population}_{ga.sigma}_{ga.n_elites}"
    session = Session(ga, name)

    session.start()


if __name__ == "__main__":
    main()
