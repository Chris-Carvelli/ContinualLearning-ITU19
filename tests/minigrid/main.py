from tests.minigrid.ga import GA
import gym_minigrid

import pickle


def main():
    print('main')

    ga = GA('MiniGrid-Empty-Noise-8x8-v0', 1000, 5,
            sigma=0.005,
            truncation=20,
            elite_trials=5,
            n_elites=1)

    pickle.dump(ga, open('ga.pickle', 'wb'))
    fp = open('process.pickle', 'ab')
    while True:
        try:
            res = ga.optimize()
            print(f'done with generation{res[0]}')
            pickle.dump(res, fp)
        except StopIteration:
            break
    fp.close()


if __name__ == "__main__":
    main()
