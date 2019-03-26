import gym
import copy
import random
import numpy as np
import src.ControllerFactory as mf
from configparser import ConfigParser
from src.ControllerFactory import *
from custom_envs import *

termination_strategies = \
    {
        'all_generations': lambda ga_instance: ga_instance.g < ga_instance.max_generations,
    }
model_library = \
    {
        'ntm': lambda copy_size: builder_ntm(copy_size),
        'base': lambda: builder_base(),
        'hyper': lambda: builder_hyper()
    }

# TODO Make elite strategies dict and parent selection strategies dict
elite_strategies = \
    {

    }
parent_selection_strategies = \
    {
    }
class GA:
    def __init__(self, config_file_path=None,
                 env_key=None,
                 population=None,
                 model_builder=None,
                 max_generations=None,
                 max_evals=None,
                 max_episode_eval=None,
                 sigma=None,
                 truncation=None,
                 trials=None,
                 elite_trials=None,
                 n_elites=None,
                 hyper_mode=None):
        config = ConfigParser()
        if config_file_path is not None:
            config.read(config_file_path)
            print("NON DEFAULT")
        else:
            config.read('config_files/config_default')
            print("DEFAULT CONFIG")

        if model_builder is None:
            self.model_builder = model_library[config["Controller"]["model_builder"]]
        else:
            self.model_builder = model_builder

        if population is None:
            self.population = int(config["HyperParameters"]["population"])
        else:
            self.population = population

        if max_episode_eval is None:
            self.max_episode_eval = int(config["HyperParameters"]["max_episode_eval"])
        else:
            self.max_episode_eval = max_episode_eval

        if max_evals is None:
            self.max_evals = int(config["HyperParameters"]["max_evals"])
        else:
            self.max_evals = max_evals

        if max_generations is None:
            self.max_generations = int(config["HyperParameters"]["max_generations"])
        else:
            self.max_generations = max_generations

        if sigma is None:
            self.sigma = float(config["HyperParameters"]["sigma"])
        else:
            self.sigma = sigma

        if truncation is None:
            self.truncation = int(config["HyperParameters"]["truncation"])
        else:
            self.truncation = truncation

        if trials is None:
            self.trials = int(config["HyperParameters"]["trials"])
        else:
            self.trials = trials

        if elite_trials is None:
            self.elite_trials = int(config["HyperParameters"]["elite_trials"])
        else:
            self.elite_trials = elite_trials

        if n_elites is None:
            self.n_elites = int(config["HyperParameters"]["n_elites"])
        else:
            self.n_elites = n_elites

        if hyper_mode is None:
            self.hyper_mode = bool(int(config["HyperParameters"]["hyper_mode"]))
        else:
            self.hyper_mode = hyper_mode

        if env_key is None:
            self.env_key = str(config["EnvironmentSettings"]["env_key"])
        else:
            self.env_key = env_key

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
    '''
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
    '''
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
        print(f'[gen {self.g}] get best Controllers')
        scored_models = self.get_best_models(self.models, self.trials)
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        print(f'[gen {self.g}] get parents')
        # self.scored_parents = self.get_best_models([m for m, _ in scored_models[:self.truncation]])
        if self.elite_trials <= 0:
            scored_parents = scored_models[:self.truncation]
        else:
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

        self.evaluations_used += sum([sum(map(lambda x: x[1], xs)) for (_, xs) in scored_models])
        scored_models = [(m, sum(map(lambda x: x[0], xs)) / float(len(xs))) for m, xs in scored_models]

        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    # @profile
    def _reproduce(self, scored_parents):
        # Elitism
        self.models = []

        for individual in range(self.population - self.n_elites):
            random_choice = random.choice(scored_parents)
            cpy = copy.deepcopy(random_choice)[0]
            self.models.append(cpy)
            self.models[-1].evolve(self.sigma)
        self.models += [p for p, _ in scored_parents[:self.n_elites]]

    def init_models(self):
        if not self.scored_parents:
            return [self.model_builder() for _ in range(self.population)]
        else:
            self._reproduce(self.scored_parents)
            return self.models

    def score_models(self, models, trials):
        ret = []
        # not pytonic but clear, check performance and memory footprint
        for m in models:
            run_res = []
            for t in range(trials):
                run_res.append(m.evaluate(self.env, self.max_episode_eval))
            ret.append((m, run_res))

        return ret

    # serialization
    def __getstate__(self):
        state = self.__dict__.copy()

        del state['models']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.models = None
        self.init_models()
