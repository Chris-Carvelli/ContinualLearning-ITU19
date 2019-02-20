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

    ga = GA('MiniGrid-Empty-Noise-8x8-v0', 100,
            max_generations=20,
            max_episode_eval=100,
            sigma=0.005,
            truncation=7,
            elite_trials=5,
            n_elites=1,
            hyper_mode=True)

    session = Session(ga, 'test_session')

    session.start()  # After running once this can be commented out


if __name__ == "__main__":
    main()
