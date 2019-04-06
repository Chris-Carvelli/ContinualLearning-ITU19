import random
import numpy
import torch

from custom_envs import *
from sessions.session import Session
from src.modules.CopyNTM import CopyNTM
from tests.minigrid.utils import lowpriority
from src.ga import GA


def main():
    lowpriority()

    data_nr = 1
    copy_size = 4
    seed = 3
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    env_key = f"CopyRnd-{copy_size}-v0"
    config = "config_ntm_copy"
    config = "config_ntm_copy_small"

    ga = GA("config_files/" + config, env_key=env_key, model_builder=lambda: CopyNTM(copy_size))
    name = f"{env_key}-{data_nr:04d}-{seed}-{config}"

    session = Session(ga, name)

    session.start()

if __name__ == "__main__":
    main()
