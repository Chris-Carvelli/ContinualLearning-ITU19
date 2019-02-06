from tests.ga import GA
import gym_minigrid


def main():
    print('main')

    ga = GA(1000, 'MiniGrid-Empty-Noise-8x8-v0')

    ga.optimize(50, 0.005, 10, elite_trials=10, n_elites=1)


if __name__ == "__main__":
    main()

