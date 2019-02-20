from tests.minigrid.ga import GA
import gym_minigrid

run_one = {
        'env_key': 'MiniGrid-Empty-Noise-8x8-v0',
        'population': 1000,
        'n_generation': 5,
        'max_eval': 100,
        'sigma': 0.05,
        'truncation': 10,
        'trials': 1,
        'elite_trials': 0,
        'n_elites': 1
    }

experiments = [run_one]

for runs in experiments:
        print('main')

        ga = GA()

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
