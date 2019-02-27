import psutil, os
import gym

from custom_envs import *
from tests.minigrid.ga import GA
from sessions.session import Session
from tests.minigrid.utils import *


def main():
    # Sets CPU usage priority to low
    lowpriority()

    ga = GA('MiniGrid-Choice3x1-color0-v0', 100, 2,
            sigma=0.005,
            max_evals=100,
            truncation=10,
            elite_trials=5,
            n_elites=1)

    # TODO: Find better name (my PC trims the last past of the name away)
    name = f"{ga.env_key}_{ga.population}_{ga.max_generations}_{ga.sigma}_{ga.truncation}_{ga.elite_trials}_{ga.n_elites}"
    session = Session(ga, name)

    session.start()     # After running once this can be commented out
    ga = session.load_results()
    plot(ga)

    env = gym.make(ga.env_key)
    parents = [p for p, _ in filter(lambda x: x[1] > 0, ga.scored_parents)]
    model = parents[0]
    while True:
        simulate(env, model, fps=4)


if __name__ == "__main__":
    main()
