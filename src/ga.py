import copy
import os
from pathlib import Path
from typing import Any, Callable

import gym
import random
import time
import configparser
import json

import numpy as np
import gym
from torch import nn

import src.ControllerFactory as mf
from configparser import ConfigParser
from src.ControllerFactory import *
from custom_envs import *
from src.utils import model_diff

from src.utils import load

# TODO move to appropriate file
sigma_strategies = {
    'constant': lambda self: self.sigma,
    'half-life-10': lambda self: self.sigma * 0.5 ** (self.g / 10.0),
    'half-life-30': lambda self: self.sigma * 0.5 ** (self.g / 30.0),
    'decay5': lambda self: self.sigma * 5 / (5 + self.g),
    'decay1': lambda self: self.sigma * 1 / (1 + self.g),
    'cyclic1000-0.01': lambda self: self.sigma * (0.01 + 1 - (self.g % 1000) / 1000),
    'linear1000-0.1': lambda self: self.sigma * (max(0.1, 1 + self.g * (0.1 - 1) / 1000)),
    'linear1000-0.01': lambda self: self.sigma * (max(0.01, 1 + self.g * (0.01 - 1) / 1000)),
    'linear10000-0.001': lambda self: self.sigma * (max(0.001, 1 + self.g * (0.001 - 1) / 10000)),
}

env_selections = {
    'random': lambda self: random.randrange(0, len(self.env_keys)),
    'sequential': lambda self: self.g // self.max_generations,
    'sequential_trial': lambda self: (self.active_env + 1) % len(self.env_keys)
}

termination_strategies = {
    'all_generations': lambda ga_instance: ga_instance.g < ga_instance.max_generations,
    'max_gen_or_reward': lambda ga: not (ga.g >= ga.max_generations or
                                         (len(ga.results) > 0 and ga.max_reward is not None
                                          and ga.results[-1][2] >= ga.max_reward)),
}


class GA:
    """
    Basic GA implementation.

    The model needs to implement the following interface:
        - evaluate(env, max_eval): evaluate the model in the given environment.
            returns total reward and evaluation used
        - evolve(sigma): evolves the model. Sigma is calculated by the GA
            (usually used as std in an additive Gaussian noise)

    """

    def __init__(self,
                 config_file=None,
                 env_keys=None,
                 population=None,
                 model_builder=None,
                 max_generations=None,
                 max_evals=None,
                 max_reward=None,
                 max_episode_eval=None,
                 sigma=None,
                 truncation=None,
                 trials=None,
                 elite_trials=None,
                 n_elites=None,
                 env_selection=None,
                 sigma_strategy=None,
                 termination_strategy=None,
                 ):

        self.config_file = config_file
        config = configparser.ConfigParser()
        default_config = Path(os.path.realpath(__file__)).parent.parent / 'config_files/config_default.ini'
        read_ok = config.read([default_config, config_file] if config_file else default_config)
        if len(read_ok) != 2:
            print("Warning: Failed to read all config files: " + str([self.config_file, default_config]))

        # hyperparams
        self.env_keys = env_keys or json.loads(config.get('EnvironmentSettings', 'env_keys'))
        if not env_keys and config.get('EnvironmentSettings', 'env_key'):
            self.env_keys.append(config.get('EnvironmentSettings', 'env_key'))
        self.population = population or int(config['HyperParameters']['population'])
        self.model_builder = model_builder or load(config['HyperParameters']['model_builder'])
        self.max_episode_eval = max_episode_eval or int(config['HyperParameters']['max_episode_eval'])
        self.max_evals = max_evals or int(config['HyperParameters']['max_evals'])
        self.max_reward = max_reward or float(config['HyperParameters']['max_reward'])
        self.max_generations = max_generations or int(config['HyperParameters']['max_generations'])
        self.sigma = sigma or float(config['HyperParameters']['sigma'])
        self.truncation = truncation or int(config['HyperParameters']['truncation'])
        self.trials = trials or int(config['HyperParameters']['trials'])
        self.elite_trials = elite_trials or int(config['HyperParameters']['elite_trials'])
        self.n_elites = n_elites or int(config['HyperParameters']['n_elites'])

        use_custom = lambda x: x and (not isinstance(x, str))

        # strategies
        if sigma_strategy and not isinstance(sigma_strategy, str):
            sigma_strategies["Custom"] = sigma_strategy
            sigma_strategy = "Custom"
        if env_selection and not isinstance(env_selection, str):
            env_selections["Custom"] = env_selection
            env_selection = "Custom"
        if termination_strategy and not isinstance(termination_strategy, str):
            termination_strategies["Custom"] = termination_strategy
            termination_strategy = "Custom"

        self.sigma_strategy_name = sigma_strategy or config['Strategies']['sigma_strategy']
        self.env_selection_name = env_selection or config['Strategies']['env_selection']
        self.termination_strategy_name = termination_strategy or config['Strategies']['termination']

        self.sigma_strategy = sigma_strategies[self.sigma_strategy_name]
        self.env_selection = env_selections[self.env_selection_name]
        self.termination_strategy = termination_strategies[self.termination_strategy_name]

        # algorithm state
        self.g = 0
        self.evaluations_used = 0
        self.results = []
        self.envs = list(map(lambda x: gym.make(x), self.env_keys))
        self.active_env = 0
        self.scored_parents = None
        self.models = None

    @property
    def env(self):
        return self.envs[self.active_env]

    def iterate(self):
        if self.termination_strategy(self):
            if self.models is None:
                self.models = self._init_models()

            ret = self.evolve_iter()
            self.g += 1
            print(f"[gen {self.g}] median_score={ret[0]}, mean_score={ret[1]}, max_score={ret[2]}")

            return ret
        else:
            raise StopIteration()

    def evolve_iter(self):
        # print(f'[gen {self.g}] get best Controllers')
        scored_models = self._get_best_models(self.models, self.trials)
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        # print(f'[gen {self.g}] get parents')
        # self.scored_parents = self.get_best_models([m for m, _ in scored_models[:self.truncation]])
        if self.elite_trials <= 0:
            scored_parents = scored_models[:self.truncation]
        else:
            scored_parents = self._get_best_models([m for m, _ in scored_models[:self.truncation]], self.elite_trials)

        self._reproduce(scored_parents)

        # print(f'[gen {self.g}] reproduce')

        # just reassigning self.scored_parents doesn't reduce the refcount, laking memory
        # buffering in a local variable, cleaning after the deepcopy and the assign the new parents
        # seems to be the only way the rogue reference doesn't appear
        if self.scored_parents is not None:
            del self.scored_parents[:]
        self.scored_parents = scored_parents

        ret = (median_score, mean_score, max_score, self.evaluations_used, self.scored_parents)
        self.results.append(ret)

        return ret

    def _get_best_models(self, models=None, trials=None):
        if models is None:
            models = self.models

        if trials is None:
            trials = self.trials

        scored_models = self._score_models(models, trials)

        self.evaluations_used += sum([sum(map(lambda x: x[1], res)) for (_, res) in scored_models])
        scored_models = [(m, sum(map(lambda x: x[0], xs)) / float(len(xs))) for m, xs in scored_models]

        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    def _reproduce(self, scored_parents):
        self.models = []
        sigma = self.sigma_strategy(self)
        for individual in range(self.population - self.n_elites):
            random_choice = random.choice(scored_parents)
            cpy = copy.deepcopy(random_choice)[0]
            self.models.append(cpy)
            self.models[-1].evolve(sigma)

        # Elitism
        self.models += [p for p, _ in scored_parents[:self.n_elites]]

    def _init_models(self):
        if not self.scored_parents:
            # TODO adapt old controllers to get obs and action spaces
            return [self.model_builder() for _ in range(self.population)]
        else:
            self._reproduce(self.scored_parents)
            return self.models

    def _score_models(self, models, trials):
        ret = []
        # not pytonic but clear, check performance and memory footprint
        for m in models:
            run_res = []
            for t in range(trials):
                self.active_env = self.env_selection(self)
                run_res.append(m.evaluate(self.envs[self.active_env], self.max_episode_eval))
            ret.append((m, run_res))

        return ret

    # serialization
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['models']

        return state

    def __setstate__(self, state):
        # Rerun init to help the allows the GA to easier unpickle successfully with older version with less
        keywords = {
            "config_file",
            "env_keys",
            "population",
            "model_builder",
            "max_generations",
            "max_evals",
            "max_reward",
            "max_episode_eval",
            "sigma",
            "truncation",
            "trials",
            "elite_trials",
            "n_elites",
            "env_selection",
            "sigma_strategy",
            "termination_strategy",
        }
        if "termination_strategy_name" in state:
            state["termination_strategy"] = state["termination_strategy_name"]
        if "env_key" in state:
            state["env_keys"] = [state["env_key"]]

        keywords_args = dict()
        for key in keywords:
            keywords_args[key] = state.get(key)

        self.__init__(**keywords_args)

        self.models = None
        self.g = state.get("g") or self.g
        self.evaluations_used = state.get("evaluations_used") or self.evaluations_used
        self.results = state.get("results") or self.results
        self.active_env = state.get("active_env") or 0
        self.scored_parents = state.get("scored_parents") or self.scored_parents
        self._init_models()

