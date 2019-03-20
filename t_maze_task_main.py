import random
import numpy
import torch

from custom_envs import *
from sessions.session import Session
from src.ControllerFactory import builder_ntm
from src.Controllers.ControllerBase import Controller
from src.modules.NTM_TMazeModule import TMazeNTMModule
from tests.minigrid.utils import lowpriority
from src.ga import GA


def main():
    lowpriority()

    seed = 3
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    length = 2
    rounds = 5
    memory_unit_size = 4

    env_key = f"TMaze-{length}x{rounds}-v0"

    ga = GA(None, env_key=env_key, population=100, model_builder=lambda: Controller(TMazeNTMModule(memory_unit_size)),
            max_generations=5000,
            sigma=0.5,
            truncation=10,
            trials=1,
            elite_trials=1,
            n_elites=5,
            )

    name = f"{env_key}-01_{ga.population}_{ga.sigma}_{ga.n_elites}_{ga.trials}-{memory_unit_size}"
    session = Session(ga, name)

    session.start()

if __name__ == "__main__":
    main()
