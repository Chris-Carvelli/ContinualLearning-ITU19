from collections import defaultdict

import psutil, os
import gym

from sessions.session import Session
from tests.minigrid.utils import plot

name = "CopyRnd-2-v0_10_100_0.5_5"
session = Session(None, name)
ga = session.load_results()
plot(ga)

gen = -1  # Last
for x in range(4):
    champ = ga.results[gen][-1][x % len(ga.results[gen][-1])][0]
    champ.history = defaultdict(list)
    res = champ.evaluate(ga.env, 100000, render=True)
    print(res)
    if hasattr(champ, "ntm"):
        champ.ntm.plot_history()

# champ = ga.results[gen][-1][-1][0]
# champ.history = defaultdict(list)
# res = evaluate_model(ga.env, champ, 100000, n=1, render=True)
# champ.plot_history()
# print(champ.memory)
