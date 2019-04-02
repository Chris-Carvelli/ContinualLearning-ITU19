import random
import sys
from collections import defaultdict

import numpy
import torch
from gym_minigrid import minigrid

from custom_envs import *
from sessions.session import Session, load_session
from src.modules import NTM_Module
from src.modules.NTM_TMazeModule import TMazeNTMModule
from src.utils import parameter_stats, model_diff
from tests.minigrid.utils import lowpriority, plot
from src.ga import GA

numpy.set_printoptions(threshold=sys.maxsize)

minigrid.AGENT_VIEW_SIZE = 3
minigrid.OBS_ARRAY_SIZE = (minigrid.AGENT_VIEW_SIZE, minigrid.AGENT_VIEW_SIZE, 3)

seed = 4

data_nr = 17
# sigma_strategy = "half-life-10"
sigma_strategy = "cyclic1000-0.01"
population = 500
sigma = .1

config = "config_ntm_default"
length = 3
rounds = 5
memory_unit_size = 2
r_inputs = 1
max_memory = 3

env_key = f"TMaze-{length}x{rounds}-v0"
name = f"{env_key}-{data_nr:04d}-{config}-{population}_{sigma}_{memory_unit_size}_{max_memory}_{minigrid.AGENT_VIEW_SIZE}_{sigma_strategy}"
# name = f"{env_key}-{data_nr:04d}-{config}-{population}_{sigma}_{memory_unit_size}_{max_memory}_{minigrid.AGENT_VIEW_SIZE}_{sigma_strategy}"


def main():
    lowpriority()
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    ga = GA("config_files/" + config,
            env_key=env_key,
            model_builder=lambda: TMazeNTMModule(memory_unit_size, max_memory=max_memory, reward_inputs=r_inputs),
            population=population,
            sigma=sigma,
            trials=2,
            elite_trials=0,
            truncation=-1,
            n_elites=10,
            sigma_strategy=sigma_strategy
            )
    session = Session(ga, name)
    session.start()


def restart():
    # ga = load_session(f"Experiments/{name}.ses")
    name = "TMaze-3x5-v0-0017-config_ntm_default-300_0.1_2_1_linear1000-0.01_RESTART"
    ga = load_session(f"Experiments/{name}.ses")
    from src.ga import sigma_strategies
    ga.sigma_strategy = sigma_strategies["cyclic1000-0.01"]

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
    # ga = load_session(f"Experiments/TMaze-1x10-v0-{data_nr:04d}-config_ntm_default-100_0.5_2.ses")
    # ga = load_session(f"Experiments/TMaze-3x5-v0-0017-config_ntm_default-300_0.1_2_1_linear1000-0.01_RESTART.ses")
    # ga = load_session(f"Experiments/TMaze-3x5-v0-0017-config_ntm_default-300_0.1_2_1_linear1000-0.01_RESTART.ses")
    ga = load_session(f"Experiments/{name}.ses")

    print("Lead generation parameter standard deviation")
    # model_diff([ga.results[-1][-1][i][0].nn for i in range(len(ga.results[-1][-1]))])
    plot(ga)
    # sys.exit()

    from custom_envs.envs import TMaze
    import numpy as np
    # env = TMaze(1, 5)
    env = ga.env

    # print(f"Champion standard deviation difference of last {min(10, len(ga.results))} generations")
    # model_diff([ga.results[i][-1][0][0].nn for i in range(min(len(ga.results), 10))])

    gen = -1  # Last
    for x in range(4):

        champ: torch.nn.Module = ga.results[gen][-1][x % len(ga.results[gen][-1])][0]
        module = champ

        if hasattr(module, "history"):
            parameter_stats(module, False)
            module.history = defaultdict(list)
        res = champ.evaluate(env, 100000, render=True, fps=6, mode="print")
        print(res)
        res = champ.evaluate(env, 100000, render=True, fps=6, mode="print")
        print(res)
        if hasattr(module, "history"):
            module.plot_history(vmin=0, vmax=1)
        if hasattr(champ, "nn") and isinstance(champ.nn, type(NTM_Module)):
            champ.nn.plot_history(vmin=0, vmax=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and "plot" in sys.argv[1]:
        plot_results()
    elif len(sys.argv) > 1 and "restart" in sys.argv[1]:
        restart()
    else:
        main()
