import os
import random
import click
import sys
from configparser import ConfigParser
from pathlib import PurePath as Path

import numpy
import torch

from custom_envs import *
from sessions.session import Session, load_session, MultiSession
from src.modules.CopyNTM import CopyNTM
from src.utils import model_diff
from tests.minigrid.utils import lowpriority, plot
from src.ga import GA


@click.command()
@click.option('--config_name', default="config_ntm_copy_2", help="Name of the config file")
@click.option('--config_folder', default="config_files/copy/", help="folder where the config file is located")
@click.option('--session_name', default=None, type=str, help='Session name. Default/None means same as config_file')
@click.option('--multi_session', default=1, help='Repeat experiment as a multi-session n times if n > 1')
def run(config_name, config_folder, session_name, multi_session):
    if session_name is None:
        session_name = config_name if multi_session <= 1 else f"{config_name}-x{multi_session}"

    # Load config
    config = ConfigParser()
    read_ok = config.read(f"{config_folder}/{config_name}")
    assert len(read_ok) > 0

    def config_get(section, key, default=None):
        if section in config and key in config[section]:
            return config[section][key]
        return default

    if config_get("Controller", "module") == "CopyNTM":
        kwargs = dict([(key, int(value)) for key, value in config["NTM"].items()])
        model_builder = lambda: CopyNTM(**kwargs)

    lowpriority()
    seed = config_get("HyperParameters", "seed")
    if seed:
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

    workers = []
    for i in range(max(1, multi_session)):
        workers.append(GA(f"{config_folder}/{config_name}", model_builder=model_builder))

    if multi_session > 1:
        session = MultiSession(workers, session_name)
    else:
        session = Session(workers[0], session_name)

    if not os.path.isfile(Path(session.save_folder) / "config"):
        fp = open(Path(session.save_folder) / "config", "a")
        config.write(fp)
    session.start()


if __name__ == '__main__':
    run()
