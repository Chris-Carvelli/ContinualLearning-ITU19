from collections import defaultdict

import psutil, os
import gym

from tests.copy.copy_experiment import MyModel
from models.ntm import CopyNTM, evaluate_model
from tests.minigrid.utils import *

copy_size = 1
length = 4
population = 500
sigma = 0.005

env_key = f"Copy-{copy_size}x{length}-v0"
name = f"{env_key}_02_{population}_{sigma}"
session = Session(None, name)
ga = session.load_results()
plot(ga)
champ = ga.results[-1][-1][0][0]
for x in range(5):
    champ.history = defaultdict(list)
    res = evaluate_model(ga.env, champ, 100000, n=1)
    print(res)
    champ.plot_history()
