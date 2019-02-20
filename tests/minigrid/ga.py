import copy
import random
import pickle
import os

from .Model import *


class GA:
    def __init__(self, env_key, population, n_generation,
                 max_eval=100,
                 sigma=0.05,
                 truncation=10,
                 trials=1,
                 elite_trials=0,
                 n_elites=1):

        # hyperparams DOING
        self.hyperparams = {}
        # hyperparams TODO create separate container class to serialize
        self.population = population
        self.env_key = env_key
        self.max_eval = max_eval
        self.n_generation = n_generation
        self.sigma = sigma
        self.truncation = truncation
        self.trials = trials
        self.elite_trials = elite_trials
        self.n_elites = n_elites

        self.scored_parents = None
        self.models = self.init_models()
        
        # strategies TODO create collections of strategies, set up externally (NO INTERNAL DICT< BAD FOR PERFORMANCE)
        self.termination_strategy = lambda: self.g < self.n_generation

        # algorithm state
        self.g = 0

    def optimize(self):
        if self.termination_strategy():
            ret = self.evolve_iter()
            self.g += 1
            return ret
        else:
            raise StopIteration()

    def evolve_iter(self):
        scored_models = self.get_best_models()
        models = [m for m, _ in scored_models]
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        self.scored_parents = self.get_best_models(models[:self.truncation])
        parents = [p for p, _ in filter(lambda x: x[1] > 0, self.scored_parents)]
        # Elitism
        self.models = parents[:self.n_elites]

        self.reproduce()

        return median_score, mean_score, max_score, self.scored_parents

    def get_best_models(self, models=None):
        if models is None:
            models = self.models

        # TODO refactor with for loops
        scored_models = list(zip(
            models,
            map(
                evaluate_model,
                [self.env_key] * (len(models) * self.trials),
                [y for x in models for y in self.trials * [x]],
                [self.max_eval] * (len(models) * self.trials))
            )
        )

        scored_models = [(scored_models[i][0], sum(s for _, s in scored_models[i * self.trials:(i + 1)*self.trials]) / self.trials)
                         for i in range(0, len(scored_models), self.trials)]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    def reproduce(self):
        for individual in range(self.population - self.n_elites):
            self.models.append(copy.deepcopy(random.choice(self.scored_parents)[0]))
            self.models[-1].evolve(self.sigma)

    def init_models(self):
        if self.scored_parents is None:
            return [Model() for _ in range(self.population)]
        else:
            self.reproduce()
            # TODO horrible, make reproduce return the models. Maintain style all over the place
            return self.models()

    # serialization
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['models']

        # TMP find better way (subObject could be ok but heavy syntax and additional ref)
        for k in self.__dict__.keys():
            if '_strategy' in k:
                state.pop(k)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        self.models = None
        self.reproduce()
