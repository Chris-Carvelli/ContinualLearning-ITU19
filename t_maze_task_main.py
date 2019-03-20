import random
from collections import defaultdict

import numpy
import torch

from custom_envs import *
from sessions.session import Session, load_session
from src.ControllerFactory import builder_ntm
from src.Controllers.ControllerNTM import Controller
from src.modules.NTM_TMazeModule import TMazeNTMModule
from tests.minigrid.utils import lowpriority, plot
from src.ga import GA



def main():
    lowpriority()
    seed = 3
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    config = "config_ntm_default"
    length = 1
    rounds = 5
    memory_unit_size = 10

    env_key = f"TMaze-{length}x{rounds}x12-v0"

    ga = GA("config_files/" + config,
            env_key=env_key,
            model_builder=lambda: Controller(TMazeNTMModule(memory_unit_size)),
            population=30,
            sigma=0.5,
            # truncation=10,
            # trials=1,
            # elite_trials=1,
            # n_elites=5,
            )
    name = f"{env_key}-0004-{config}-{ga.population}_{ga.sigma}_{memory_unit_size}"

    session = Session(ga, name)
    session.start()


def plot_results():
    # ga = load_session("Experiments/TMaze-1x5x20-v0-0004-config_ntm_default-100_0.5_1.ses")
    ga = load_session("Experiments/TMaze-1x5x12-v0-0004-config_ntm_default-30_0.5_10.ses")
    plot(ga)

    gen = -1  # Last
    for x in range(4):
        champ = ga.results[gen][-1][x % len(ga.results[gen][-1])][0]
        if hasattr(champ, "ntm"):
            champ.ntm.history = defaultdict(list)
        res = champ.evaluate(ga.env, 100000, render=True, fps=4)
        print(res)
        if hasattr(champ, "ntm"):
            champ.ntm.plot_history()


if __name__ == "__main__":
    main()
    # plot_results()
