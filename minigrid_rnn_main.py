import random
import sys
from collections import defaultdict

import gym
import numpy
import torch
from gym_minigrid import minigrid

from custom_envs import *
from sessions.session import Session, load_session
from src.modules.MinigridRecurrentModule import MinigridRNNModule
from src.utils import parameter_stats, model_diff
from tests.minigrid.utils import lowpriority, plot
from src.ga import GA
import utils

numpy.set_printoptions(threshold=sys.maxsize, linewidth=200)

seed = 0
data_nr = 1
# config = "config_minigrid_rnn_fast"
config = "config_minigrid_rnn"
env_key = f"MiniGrid-DoorKey-5x5-v0"
name = f"{env_key}-{data_nr:04d}-{seed}-{config}"


def main():
    lowpriority()
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    ga = GA("config_files/" + config,
            env_key=env_key,
            model_builder=lambda: MinigridRNNModule(env_key),
            )
    print(ga.max_reward)
    session = Session(ga, name)
    session.start()



def plot_results():
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
        res = champ.evaluate(env, 100000, render=True, fps=6)
        print(res)


if __name__ == "__main__":
    if len(sys.argv) > 1 and "plot" in sys.argv[1]:
        plot_results()
    else:
        main()

