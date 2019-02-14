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

    ga = GA('MiniGrid-Choice3x1-color0-v0', 100, 2,
            sigma=0.005,
            truncation=10,
            elite_trials=5,
            n_elites=1)

    # TODO: Find better name (my PC trims the last past of the name away)
    name = f"{ga.env_key}_{ga.population}_{ga.n_generation}_{ga.sigma}_{ga.truncation}_{ga.elite_trials}_{ga.n_elites}"
    session = Session(ga, name)

    # session.start()
    plot(name)
    ga = session.load_results()

    env = gym.make(ga.env_key)
    parents = [p for p, _ in filter(lambda x: x[1] > 0, ga.scored_parents)]
    model = parents[0]
    while True:
        simulate(env, model, fps=4)


if __name__ == "__main__":
    main()
