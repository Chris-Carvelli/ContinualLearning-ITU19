import psutil, os
import gym_minigrid

from custom_envs import *
from tests.minigrid.ga import GA
from sessions.session import Session
from tests.minigrid.utils import plot, simulate, lowpriority

import pickle

def main():
    print('main')

    # Sets CPU usage priority to low
    lowpriority()

    ga = GA('Frostbite-v4', 1000,
            max_evals=6_000_000_000,
            max_episode_eval=5000,
            sigma=0.002,
            truncation=20,
            elite_trials=30,
            n_elites=1,
            hyper_mode=True)

    while True:
        try:
            ga.iterate()
        except StopIteration:
            break


if __name__ == "__main__":
    main()
