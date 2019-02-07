import copy
import random
import pickle
import os

from tests.frozenLake_chris.Model import *


class GA:
    def __init__(self, population, env_key):
        self.population = population
        self.env_key = env_key

        self.models = [Model()]

    def get_best_models(self):
        scored_models = list(zip(map(evaluate_model, [self.env_key], self.models), self.models))
        scored_models.sort(key=lambda x: x[0], reverse=True)

        return scored_models

    def evolve_iter(self, sigma, truncation):
        scored_models = self.get_best_models()
        scores = [s for s, _ in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        parents = scored_models[truncation]
        # Elitism
        self.models = [parents[0]]

        for _ in range(self.population):
            self.models.append(copy.deepcopy(random.choice(scored_models)[0]))
            self.models[-1].evolve(sigma)

        return median_score, mean_score, max_score, self.models[0]

    def optimize(self, n_generation, sigma, truncation):
        print('start')
        path = os.path.join(
            os.getcwd(),
            'Experiments',
            self.env_key,
            f'{self.population}_{n_generation}_{sigma}_{truncation}')
        os.makedirs(path)
        fp = open(f'{path}\\process.pickle', 'ab')

        for g in range(n_generation):
            med, avg, M, elite = self.evolve_iter(sigma, truncation)

            print(f'Done with generation {g} [Median: {med}, average: {avg}, max: {M}]')
            pickle.dump((med, avg, M, elite), fp)

        print('done')

