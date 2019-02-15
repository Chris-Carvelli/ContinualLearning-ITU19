import copy
import random
import pickle
import os, sys
from memory_profiler import profile
import gym
from .Model import *


class GA:
    def __init__(self, env_key, population, n_generation,
                 max_eval=100,
                 sigma=0.05,
                 truncation=10,
                 trials=1,
                 elite_trials=0,
                 n_elites=1):

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

        # strategies TODO create collections of strategies, set up externally (NO INTERNAL DICT< BAD FOR PERFORMANCE)
        self.termination_strategy = lambda: self.g < self.n_generation

        # algorithm state
        self.g = 0
        self.scored_parents = None
        self.models = None
        self.env = gym.make(env_key)

        # results TMP check if needed
        self.med_scores = []
        self.avg_scores = []
        self.max_scores = []

        self.models = self.init_models()

    def iterate(self):     # Renamed
        # if self.g == 0:
        #     self.models = self.init_models()
        if self.termination_strategy():
            ret = self.evolve_iter()
            self.g += 1
            print(f"median_score={ret[0]}, mean_score={ret[1]}, max_score={ret[2]}")

            return ret
        else:
            raise StopIteration()

    def evolve_iter(self):
        print(f'[gen {self.g}] get best models')
        scored_models = self.get_best_models()
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        print(f'[gen {self.g}] get parents')
        self.scored_parents = self.get_best_models([m for m, _ in scored_models[:self.truncation]])
        # del scored_models

        print(f'[gen {self.g}] reproduce')
        self.reproduce()
        self.med_scores.append(median_score)
        self.avg_scores.append(mean_score)
        self.max_scores.append(max_score)

        return median_score, mean_score, max_score, self.scored_parents

    def get_best_models(self, models=None):
        if models is None:
            models = self.models

        # TODO refactor with for loops
        scored_models = list(zip(
            models,
            map(
                evaluate_model,
                [self.env] * (len(models) * self.trials),
                [y for x in models for y in self.trials * [x]],
                [self.max_eval] * (len(models) * self.trials))
            )
        )

        scored_models = [(scored_models[i][0], sum(s for _, s in scored_models[i * self.trials:(i + 1)*self.trials]) / self.trials)
                         for i in range(0, len(scored_models), self.trials)]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    # @profile
    def reproduce(self):
        elites = [m for m, _ in self.scored_parents[:self.n_elites]]
        for model in self.models:
            if model not in elites:
                model.become_child(random.choice(self.scored_parents)[0])
                model.evolve(self.sigma)
            # else:
            #     print('elite')

        # TMP profiling
        # TMP_generator = range(self.population - self.n_elites)
        # for individual in TMP_generator:
        #     random_choice = random.choice(self.scored_parents)
        #     cpy = copy.deepcopy(random_choice)[0]
        #     self.models.append(cpy)
        #     self.models[-1].evolve(self.sigma)

    def init_models(self):
        if not self.scored_parents:
            return [Model() for _ in range(self.population)]
        else:
            self.reproduce()
            # TODO horrible, make reproduce return the models. Maintain style all over the place
            return self.models

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

        self.models = None
        self.init_models()
