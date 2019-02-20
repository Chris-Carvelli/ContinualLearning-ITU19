from tests.minigrid.ga import GA
import gym_minigrid


elite_strategies = \
    {

    }

termination_strategies = \
    {
        "end": lambda: self.g < self.max_generations
    }


class Config:
    def __init__(self, env_key, population,
                 max_generations=20,
                 max_evals=1000,
                 max_episode_eval=100,
                 sigma=0.05,
                 truncation=10,
                 trials=1,
                 elite_trials=0,
                 n_elites=1,
                 hyper_mode=True):
        self.env_key = env_key
        self.population = population
        self.max_generations = max_generations
        self.max_evals = max_evals
        self.max_episode_eval = max_episode_eval
        self.sigma = sigma
        self.truncation = truncation
        self.trials = trials
        self.elite_trials = elite_trials
        self.n_elites = n_elites
        self.hyper_mode = hyper_mode


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
