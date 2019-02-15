import psutil, os
import gym
import gym_minigrid

from custom_envs import *
from tests.minigrid.ga import GA
from sessions.session import Session
from tests.minigrid.utils import plot, simulate, lowpriority


def main():
    print('main')

    # Sets CPU usage priority to low
    lowpriority()

    ga = GA('MiniGrid-Empty-6x6-v0', 50, 10,
            sigma=0.005,
            truncation=5,
            elite_trials=5,
            n_elites=1)

    # # TODO: Find better name (my PC trims the last past of the name away)
    # name = f"{ga.env_key}_{ga.population}_{ga.n_generation}_{ga.sigma}_{ga.truncation}_{ga.elite_trials}_{ga.n_elites}"
    # session = Session(ga, name)
    # session.start()

    while True:
        try:
            ga.iterate()
        except StopIteration:
            break



if __name__ == "__main__":
    main()
