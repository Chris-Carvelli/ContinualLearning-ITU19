import random
import numpy
import torch

from custom_envs import *
from sessions.session import Session
from src.ModelFactory import builder_ntm
from tests.minigrid.utils import lowpriority
from src.ga import GA


def main():
    lowpriority()

    copy_size = 2
    seed = 3
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    env_key = f"CopyRnd-{copy_size}-v0"

    ga = GA(env_key, 100, model_builder=lambda: builder_ntm(copy_size),
            max_generations=5000,
            sigma=0.5,
            truncation=10,
            trials=50,
            elite_trials=50,
            n_elites=5,
            )

    name = f"{env_key}-01_{ga.population}_{ga.sigma}_{ga.n_elites}"
    session = Session(ga, name)

    session.start()

if __name__ == "__main__":
    main()
