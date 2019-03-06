import random

import torch

import gym

# TMP specify the wanted builder in the config file
from src.ModelFactory import builder_hyper as builder


def random_state():
    return random.randint(0, 2**31-1)


class CompressedModel:
    def __init__(self, start_rng=None, other_rng=None):
        self.start_rng = start_rng if start_rng is not None else random_state()
        self.other_rng = other_rng if other_rng is not None else []

    def evolve(self, sigma, rng_state=None):
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))


def worker_evaluate_model(env_key, compressed_model, max_eval):
    env = gym.make(env_key)
    model = uncompress_model(compressed_model)

    return model.evaluate(env, max_eval)


def uncompress_model(compressed_model):
    start_rng, other_rng = compressed_model.start_rng, compressed_model.other_rng

    torch.manual_seed(start_rng)
    m = builder()
    for sigma, rng in other_rng:
        torch.manual_seed(rng)
        m.evolve(sigma)

    return m
