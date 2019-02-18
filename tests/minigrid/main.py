from tests.minigrid.ga import GA
import gym_minigrid
import custom_envs

import pickle


def main():
    print('main')

    ga = GA('Frostbite-v4', 1000,
            max_evals=6_000_000_000,
            sigma=0.002,
            truncation=20,
            elite_trials=30,
            n_elites=1,
            hyper_mode=True)

    pickle.dump(ga, open('ga.pkl', 'wb'))
    fp = open('process.pkl', 'wb')
    g = 0
    while True:
        try:
            res = ga.optimize()
            print(f'done with generation {g}: {res[:4]}')
            g += 0
            pickle.dump(res, fp)
            pickle.dump(ga, open('ga.pkl', 'wb'))
        except StopIteration:
            break
    fp.close()


if __name__ == "__main__":
    main()
