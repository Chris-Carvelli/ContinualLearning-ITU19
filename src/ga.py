import copy
import random
import time
import configparser
import json

import numpy as np
import gym

from Logger import Logger
from utils import load


class GA:
    # TMP move to appropriate file
    env_selections = None

    """
    Basic GA implementation.

    The model needs to implement the following interface:
        - evaluate(env, max_eval): evaluate the model in the given environment.
            returns total reward and evaluation used
        - evolve(sigma): evolves the model. Sigma is calculated by the GA
            (usually used as std in an additive Gaussian noise)

    """

    def __init__(self,
                 envs=None,
                 env_selection=None,
                 config_file='ga',
                 population=None,
                 model_builder=None,
                 max_generations=None,
                 max_evals=None,
                 max_episode_eval=None,
                 sigma=None,
                 sigma_decay=None,
                 min_sigma=None,
                 truncation=None,
                 trials=None,
                 elite_trials=None,
                 n_elites=None,
                 graphical_output=False):

        config = configparser.ConfigParser()
        config.read(f'../experiments/{config_file}.cfg')

        self.config_file = config_file

        self.env_selections = {
            'random': lambda: random.randrange(0, len(self.env_keys)),
            'sequential': lambda: self.g // self.max_generations,
            'sequential_trial': lambda: (self.active_env + 1) % len(self.env_keys)
        }

        # hyperparams
        self.env_keys = envs or json.loads(config.get('ENVS', 'keys'))
        k = env_selection or config.get('ENVS', 'env_selection')
        self.env_selection = self.env_selections[k]
        self.population = population or int(config['HYPERPARAMS']['population'])
        self.model_builder = model_builder or load(config['HYPERPARAMS']['model_builder'])
        self.max_episode_eval = max_episode_eval or int(config['HYPERPARAMS']['max_episode_eval'])
        self.max_evals = max_evals or int(config['HYPERPARAMS']['max_evals'])
        self.max_generations = max_generations or int(config['HYPERPARAMS']['max_generations'])
        self.sigma = sigma or float(config['HYPERPARAMS']['sigma'])
        self.sigma_decay = sigma_decay or float(config['HYPERPARAMS']['sigma_decay'])
        self.min_sigma = min_sigma or float(config['HYPERPARAMS']['min_sigma'])
        self.truncation = truncation or int(config['HYPERPARAMS']['truncation'])
        self.trials = trials or int(config['HYPERPARAMS']['trials'])
        self.elite_trials = elite_trials or int(config['HYPERPARAMS']['elite_trials'])
        self.n_elites = n_elites or int(config['HYPERPARAMS']['n_elites'])

        self.envs = list(map(lambda x: gym.make(x), self.env_keys))
        self.active_env = 0

        # population
        self.scored_parents = None
        self.models = None

        # strategies
        self.termination_strategy = lambda: \
            self.g < self.max_generations * len(self.envs) and \
            self.evaluations_used < self.max_evals

        # algorithm state
        self.g = 0
        self.evaluations_used = 0

        # utils
        self.graphical_output = graphical_output

    def optimize(self):
        """
        Runs a generation of the GA. The result is a tuple
          (median_score, mean_score, max_score, evaluation_used, scored_parents)

        :return: False if the optimization is ended, result otherwise
        """
        if self.termination_strategy():
            if self.models is None:
                self._log(f'{"Res" if self.g > 0 else "S"}tarting run')
                self.models = self._init_models()

            self._log(f'start gen {self.g}')
            ret = self._evolve_iter()
            self.g += 1
            self._log(f"median_score={ret[0]}, mean_score={ret[1]}, max_score={ret[2]}")

            return ret
        else:
            self._log('end')
            return False

    def _evolve_iter(self):
        scored_models = self._get_best_models(self.models, self.trials, 'Score Population')
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        if self.elite_trials > 0:
            scored_parents = self._get_best_models(
                [m for m, _ in scored_models[:self.truncation]], self.elite_trials, 'Score elite'
            )
        else:
            scored_parents = scored_models[:self.truncation]

        self._reproduce(scored_parents)

        self._log(f'[gen {self.g}] reproduce')

        # just reassigning self.scored_parents doesn't reduce the refcount, laking memory
        # buffering in a local variable, cleaning after the deepcopy and the assign the new parents
        # seems to be the only way the rogue reference doesn't appear
        if self.scored_parents is not None:
            del self.scored_parents[:]
        self.scored_parents = scored_parents

        ret = (median_score, mean_score, max_score, self.evaluations_used, self.scored_parents)

        return ret

    def _get_best_models(self, models=None, trials=None, queue_name='Get Best Models'):
        if models is None:
            models = self.models

        if trials is None:
            trials = self.trials

        scored_models = self._score_models(models, trials, queue_name)

        self.evaluations_used += sum(sum(map(lambda x: x[1], res)) for _, res in scored_models)
        scored_models = [(m, sum(map(lambda x: x[0], scores)) / trials) for m, scores in scored_models]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    # @profile
    def _reproduce(self, scored_parents):
        # Elitism
        self.models = [p for p, _ in scored_parents[:self.n_elites]]
        sigma = max(self.min_sigma, self.sigma * pow(self.sigma_decay, self.g))
        self._log(f'Reproduce with sigma {sigma}')

        with Logger(total=self.truncation, desc=f'Reproduce[s={sigma}]') as bar:
            for individual in range(self.population - self.n_elites):
                random_choice = random.choice(scored_parents)
                cpy = copy.deepcopy(random_choice)[0]
                self.models.append(cpy)
                self.models[-1].evolve(sigma)
                bar.step()

    def _init_models(self):
        self._log('Init models')
        if not self.scored_parents:
            # TODO adapt old controllers to get obs and action spaces
            return [self.model_builder(
                {"image": self.envs[self.active_env].observation_space.spaces['image'].shape, "text": 100},
                self.envs[self.active_env].action_space)
                for _ in range(self.population)]
        else:
            self._reproduce(self.scored_parents)
            return self.models

    def _score_models(self, models, trials, queue_name):
        ret = []
        # not pytonic but clear, check performance and memory footprint
        evaluation = 0
        with Logger(total=len(models) * trials, desc=queue_name) as bar:
            for m in models:
                m_scores = []
                for _ in range(trials):
                    self.active_env = self.env_selection()
                    m_scores.append(m.evaluate(self.envs[self.active_env], self.max_episode_eval))
                    evaluation += 1
                    bar.step(m_scores[-1])
                ret.append((m, m_scores))

            return ret

    # utils
    def _log(self, s):
        t = time.localtime()
        t_str = time.strftime("%a, %d %b %Y %H:%M:%S", t)
        with open('../results/ga.log', 'a+') as f:
            f.write(f'[{t_str}] {s}\n')

        if not self.graphical_output:
            print(f'[{self.config_file}]{s}')

    # serialization
    def __getstate__(self):
        state = self.__dict__.copy()

        del state['models']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.models = None
