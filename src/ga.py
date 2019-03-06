import gym
import copy
import random
import gym_minigrid
import numpy as np
import src.ModelFactory as mf
from configparser import ConfigParser
import os

# TMP networking
# from redis import Redis
# from rq import Queue
# import time
# from src.CompressedModel import worker_evaluate_model

# TMP
# from settings import REDIS_HOST
# from memory_profiler import profile

termination_strategies = \
    {
        'all_generations': lambda ga_instance: ga_instance.g < ga_instance.max_generations,
    }

# TODO Make elite strategies dict and parent selection strategies dict
elite_strategies =\
    {

    }
parent_selection_strategies =\
    {
    }

class GA:
    def __init__(self, env_key, population, model_builder,
                 networked=False,
                 max_generations=20,
                 max_evals=1000,
                 max_episode_eval=100,
                 sigma=0.05,
                 truncation=10,
                 trials=1,
                 elite_trials=0,
                 n_elites=1,
                 hyper_mode=True):
        self.networked = networked

        # hyperparams
        self.population = population
        self.model_builder = model_builder
        self.env_key = env_key
        self.max_episode_eval = max_episode_eval
        self.max_evals = max_evals
        self.max_generations = max_generations
        self.sigma = sigma
        self.truncation = truncation
        self.trials = trials
        self.elite_trials = elite_trials
        self.n_elites = n_elites
        self.hyper_mode = hyper_mode

        self.scored_parents = None
        self.models = None
        self.config_file = None
        # strategies
        self.termination_strategy = lambda: self.g < self.max_generations
        # self.termination_strategy = lambda: self.evaluations_used < self.max_episode_eval

        # algorithm state
        self.g = 0
        self.evaluations_used = 0
        self.env = gym.make(self.env_key)

        self.results = []


        # network management TODO move in different class
        # tmp
        # self.redis = Redis(REDIS_HOST)
        # self.queue = Queue(connection=self.redis, name='default')
        # for j in self.queue.jobs:
        #   j.cancel()

    def __init__(self, config_file_path):
        config = ConfigParser()
        config.read(config_file_path)
        self.config_file = config
        self.model_builder = mf.builder_base
        self.networked = bool(int(config["Utility"]["networked"]))
        self.population = int(config["HyperParameters"]["population"])
        self.max_episode_eval = int(config["HyperParameters"]["max_episode_eval"])
        self.max_evals = int(config["HyperParameters"]["max_evals"])
        self.max_generations = int(config["HyperParameters"]["max_generations"])
        self.sigma = float(config["HyperParameters"]["sigma"])
        self.truncation = int(config["HyperParameters"]["truncation"])
        self.trials = int(config["HyperParameters"]["trials"])
        self.elite_trials = int(config["HyperParameters"]["elite_trials"])
        self.n_elites = int(config["HyperParameters"]["n_elites"])
        self.hyper_mode = bool(int(config["HyperParameters"]["hyper_mode"]))

        self.env_key = str(config["EnvironmentSettings"]["env_key"])

        print(self.env_key)

        self.scored_parents = None
        self.models = self.init_models()

        self.termination_strategy_name = config["Strategies"]["termination"]
        # TODO Make Other strategies to be modular


        # algorithm state
        self.g = 0
        self.evaluations_used = 0
        self.env = gym.make(self.env_key)

        # results TMP check if needed
        self.results = []

        # network management TODO move in different class
        # tmp
        # self.redis = Redis(REDIS_HOST)
        # self.queue = Queue(connection=self.redis, name='default')
        # for j in self.queue.jobs:
        #   j.cancel()

    def iterate(self):
        if termination_strategies[self.termination_strategy_name](self):
            if self.models is None:
                self.models = self.init_models()

            ret = self.evolve_iter()
            self.g += 1
            print(f"median_score={ret[0]}, mean_score={ret[1]}, max_score={ret[2]}")

            return ret
        else:
            raise StopIteration()

    # @profile
    def evolve_iter(self):
        print(f'[gen {self.g}] get best models')
        scored_models = self.get_best_models(self.models, self.trials)
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        print(f'[gen {self.g}] get parents')
        # self.scored_parents = self.get_best_models([m for m, _ in scored_models[:self.truncation]])
        scored_parents = self.get_best_models([m for m, _ in scored_models[:self.truncation]], self.elite_trials)

        self._reproduce(scored_parents)

        print(f'[gen {self.g}] reproduce')

        # just reassigning self.scored_parents doesn't reduce the refcount, laking memory
        # buffering in a local variable, cleaning after the deepcopy and the assign the new parents
        # seems to be the only way the rogue reference doesn't appear
        if self.scored_parents is not None:
            del self.scored_parents[:]
        self.scored_parents = scored_parents

        ret = (median_score, mean_score, max_score, self.evaluations_used, self.scored_parents)
        self.results.append(ret)

        return ret

    def get_best_models(self, models=None, trials=None):
        if models is None:
            models = self.models

        if trials is None:
            trials = self.trials

        scored_models = self.score_models(models, trials)

        self.evaluations_used += sum(s[1] for _, s in scored_models)
        scored_models = [(scored_models[i][0], sum(s[0] for _, s in scored_models[i * trials:(i + 1)*trials]) / trials)
                         for i in range(0, len(scored_models), self.trials)]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    # @profile
    def _reproduce(self, scored_parents):
        # Elitism
        self.models = [p for p, _ in scored_parents[:self.n_elites]]

        for individual in range(self.population - self.n_elites):
            random_choice = random.choice(scored_parents)
            cpy = copy.deepcopy(random_choice)[0]
            self.models.append(cpy)
            self.models[-1].evolve(self.sigma)

    def init_models(self):
        if not self.scored_parents:
            return [self.model_builder() for _ in range(self.population)]
        else:
            self._reproduce(self.scored_parents)
            return self.models

    def score_models(self, models, trials):
        return self.network_eval(models, trials) if self.networked else self.local_eval(models, trials)

    def local_eval(self, models, trials):
        ret = []
        # not pytonic but clear, check performance and memory footprint
        for m in models:
            for t in range(trials):
                ret.append((m, m.evaluate(self.env, self.max_episode_eval)))

        return ret

    def network_eval(self, models, trials):
        print("UNIMPLEMENTED")
        # def enqueue(m):
        #     return self.queue.enqueue(
        #         worker_evaluate_model,
        #         self.env_key,
        #         m,
        #         max_eval=trials,
        #         ttl=650,
        #         timeout=600
        #     )
        #
        # jobs = []
        # for m in models:
        #     jobs.append(enqueue(m))
        # last_enqueue_time = time.time()
        # while True:
        #     for i in range(len(jobs)):
        #         if jobs[i].result is not None and not isinstance(jobs[i], FakeJob):
        #             jobs[i] = FakeJob(jobs[i])
        #
        #     def check_res(j):
        #         if j.result is not None:
        #             return j.result
        #         return None
        #
        #     scores = [check_res(j) for j in jobs]
        #     if None not in scores:
        #         break
        #     if time.time() - last_enqueue_time > 600:
        #         print(f'Reenqueuing unfinished jobs ({sum(x is None for x in scores)}).')
        #         for i in range(len(jobs)):
        #             if jobs[i].result is None:
        #                 jobs[i].cancel()
        #                 jobs[i] = enqueue(self.models[i])
        #         last_enqueue_time = time.time()
        #     time.sleep(1)
        #
        # scored_models = list(zip(self.models, scores))
        # scored_models.sort(key=lambda x: x[1], reverse=True)
        # return scored_models

    # serialization
    def __getstate__(self):
        state = self.__dict__.copy()

        del state['models']

        # TMP networking
        # del state['redis']
        # del state['queue']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.models = None
        self.init_models()


# TMP networking
class FakeJob:
    def __init__(self, j):
        self.result = j.result