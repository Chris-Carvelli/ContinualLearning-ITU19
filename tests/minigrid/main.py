import psutil, os
import gym

from custom_envs import *
from tests.minigrid.ga import GA
from sessions.session import Session
from tests.minigrid.utils import plot, simulate


def main():
    print('main')

    # Sets cpu priority below normal
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    ga = GA('MiniGrid-Empty-Noise-8x8-v0', 10, 5,
            sigma=0.005,
            truncation=20,
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
