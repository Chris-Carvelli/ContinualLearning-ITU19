import random
import sys
from configparser import ConfigParser

import numpy
import torch

from custom_envs import *
from sessions.session import Session, load_session
from src.modules.CopyNTM import CopyNTM
from src.utils import model_diff
from tests.minigrid.utils import lowpriority, plot
from src.ga import GA


def main():
    lowpriority()

    config_name = "config_ntm_copy_2"
    config_file = "config_files/copy/" + config_name
    config = ConfigParser()
    read_ok = config.read(config_file)
    assert len(read_ok) > 0

    memory_unit_size = int(config["NTM"]["memory_unit_size"])
    copy_size = int(config["NTM"]["copy_size"])
    data_nr = 1
    seed = 4
    env_key = f"CopyRnd-{copy_size}-v0"

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    ga = GA(config_file, env_key=env_key, population=500,
            model_builder=lambda: CopyNTM(copy_size, memory_unit_size=memory_unit_size))
    name = f"{env_key}-{data_nr:04d}-{seed}-{config_name}-{ga.population}"

    session = Session(ga, name)

    session.start()


def plot_results():
    name = "CopyRnd-8-v0-0001-3-config_ntm_copy-500"
    ga = load_session(f"Experiments/{name}.ses")

    print("Lead generation scores")
    print([ga.results[-1][-1][i][1] for i in range(len(ga.results[-1][-1]))])
    print("Lead generation parameter standard deviation")
    model_diff([ga.results[-1][-1][i][0].nn for i in range(len(ga.results[-1][-1]))])
    plot(ga)
    env = ga.env

    gen = -1  # Last
    for x in range(4):

        champ: torch.nn.Module = ga.results[gen][-1][x % len(ga.results[gen][-1])][0]
        module = champ

        if hasattr(module, "history"):
            module.start_history()
        res = champ.evaluate(env, 100000, render=True, fps=6)
        print(res)
        if hasattr(module, "history"):
            module.plot_history(vmin=0, vmax=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and "plot" in sys.argv[1]:
        plot_results()
    else:
        main()
