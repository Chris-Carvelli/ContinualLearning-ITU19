import random
import sys
from collections import defaultdict

import numpy
import torch

from custom_envs import *
from sessions.session import Session, load_session
from src.ControllerFactory import builder_ntm
from src.Controllers.ControllerNTM import Controller
from src.modules.NTM_TMazeModule import TMazeNTMModule
from src.utils import parameter_stats, model_diff
from tests.minigrid.utils import lowpriority, plot
from src.ga import GA



def main():
    lowpriority()
    seed = 4
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    config = "config_ntm_default"
    length = 1
    rounds = 5
    memory_unit_size = 2
    r_inputs = 1

    env_key = f"TMaze-{length}x{rounds}-v0"

    ga = GA("config_files/" + config,
            env_key=env_key,
            model_builder=lambda: Controller(TMazeNTMModule(memory_unit_size, reward_inputs=r_inputs)),
            population=100,
            sigma=0.05,
            trials=2,
            elite_trials=None,
            # truncation=10,
            n_elites=10,
            )
    # name = f"{env_key}-0007-{config}-{ga.population}_{ga.sigma}_{memory_unit_size}_r-inputs_{r_inputs}"
    name = f"{env_key}-0011-{config}-{ga.population}_{ga.sigma}_{memory_unit_size}"

    session = Session(ga, name)
    session.start()


def plot_results():
    # ga = load_session("Experiments/TMaze-1x5x20-v0-0004-config_ntm_default-100_0.5_1.ses")
    # ga = load_session("Experiments/TMaze-1x5x12-v0-0004-config_ntm_default-30_0.5_10.ses")
    # ga = load_session("Experiments/TMaze-1x5-v0-0005-config_ntm_default-200_0.5_2.ses")
    # ga = load_session("Experiments/TMaze-1x5-v0-0005-config_ntm_default-100_0.5_2.ses")
    # ga = load_session("Experiments/TMaze-1x5-v0-0007-config_ntm_default-100_0.5_2.ses")
    # ga = load_session("Experiments/TMaze-1x5-v0-0007-config_ntm_default-100_0.5_2_r-inputs_6.ses")
    # ga = load_session("Experiments/TMaze-1x5-v0-0009-config_ntm_default-300_0.5_2.ses")
    ga = load_session("Experiments/TMaze-1x5-v0-0011-config_ntm_default-100_0.005_2.ses")
    plot(ga)

    from custom_envs.envs import TMaze
    import numpy as np
    # env = TMaze(1, 3)
    env = ga.env

    print("Lead generation parameter standard deviation")
    model_diff([ga.results[-1][-1][i][0].ntm for i in range(len(ga.results[-1][-1]))])

    # print(f"Champion standard deviation difference of last {min(10, len(ga.results))} generations")
    # model_diff([ga.results[i][-1][0][0].ntm for i in range(min(len(ga.results), 10))])


    gen = -1  # Last
    for x in range(4):

        champ: torch.nn.Module = ga.results[gen][-1][x % len(ga.results[gen][-1])][0]

        if hasattr(champ, "ntm"):
            parameter_stats(champ.ntm, False)
            champ.ntm.history = defaultdict(list)
        res = champ.ntm.evaluate(env, 100000, render=True, fps=6, mode="print")
        print(res)
        if hasattr(champ, "ntm"):
            champ.ntm.plot_history(vmin=0, vmax=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and "plot" in sys.argv[1]:
        plot_results()
    else:
        main()

