import copy
import random
import pickle
import os

from .Model import *


class GA:
    def __init__(self, population, env_key, max_eval=100):
        self.population = population
        self.env_key = env_key
        self.max_eval = max_eval

        # TODO make local variable, create self.parents
        self.models = [Model() for _ in range(population)]

    def get_best_models(self, models=None, trials=1):
        if models is None:
            models = self.models

        # TODO refactor with for loops
        scored_models = list(zip(
            models,
            map(
                evaluate_model,
                [self.env_key] * (len(models) * trials),
                [y for x in models for y in trials * [x]],
                [self.max_eval] * (len(models) * trials))
            )
        )

        scored_models = [(scored_models[i][0], sum(s for _, s in scored_models[i * trials:(i + 1)*trials]) / trials)
                         for i in range(0, len(scored_models), trials)]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    def evolve_iter(self, sigma, truncation, trials, elite_trials, n_elites):
        scored_models = self.get_best_models(trials=trials)
        models = [m for m, _ in scored_models]
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        scored_parents = self.get_best_models(models[:truncation], elite_trials)
        parents = [p for p, _ in filter(lambda x: x[1] > 0, scored_parents)]
        # Elitism
        self.models = parents[:n_elites]

        for individual in range(self.population - n_elites):
            self.models.append(copy.deepcopy(random.choice(scored_models)[0]))
            self.models[-1].evolve(sigma)

        return median_score, mean_score, max_score, self.models[0]

    # TODO replace with __next__, promote all params to class members
    def optimize(self, n_generation, sigma, truncation, trials=1, elite_trials=1, n_elites=1):
        print('start')
        path = os.path.join(
            os.getcwd(),
            'Experiments',
            self.env_key,
            f'{self.population}_{n_generation}_{sigma}_{truncation}')
        os.makedirs(path)
        fp = open(f'{path}\\process.pickle', 'ab')

        for g in range(n_generation):
            s_med, s_avg, s_max, elite = self.evolve_iter(sigma, truncation, trials, elite_trials, n_elites)

            print(f'Done with generation {g} [Median: {s_med}, average: {s_avg}, max: {s_max}]')
            pickle.dump((s_med, s_avg, s_max, elite), fp)

        print('done')
