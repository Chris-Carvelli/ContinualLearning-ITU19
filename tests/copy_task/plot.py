from collections import defaultdict

import psutil, os
import gym

from tests.copy_task.copy_experiment import MyModel
from models.ntm import CopyNTM, evaluate_model
from tests.minigrid.utils import *

copy_size = 2
length = 4
# population = 100
population = 1000
sigma = 0.5
# sigma = 0.1
# sigma = 0.05
# sigma = 0.01
# sigma = 0.005
n_elites = 5

# copy_size = 8
# length = 12
# population = 100
# sigma = 0.01
# n_elites = 10


# env_key = f"Copy-{copy_size}x{length}-v0"
env_key = f"CopyRnd-{copy_size}-v0"

name = f"{env_key}_07_{population}_{sigma}_{n_elites}"
session = Session(None, name)
ga = session.load_results()
plot(ga)
# print(ga.results[-1][-1])
gen = -1  # Last
for x in range(4):
    champ = ga.results[gen][-1][x % len(ga.results[gen][-1])][0]
    champ.history = defaultdict(list)
    res = evaluate_model(ga.env, champ, 100000, n=1, use_seed=False)
    print(res)
    champ.plot_history()

# champ = ga.results[gen][-1][-1][0]
# champ.history = defaultdict(list)
# res = evaluate_model(ga.env, champ, 100000, n=1, render=True)
# champ.plot_history()
# print(champ.memory)
