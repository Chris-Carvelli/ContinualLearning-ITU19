import os
import random
import click
import sys
from configparser import ConfigParser
from pathlib import Path

import numpy
import torch

import data_analyzer as da
import seaborn as sns
import matplotlib.pyplot as plt

from custom_envs import *
from sessions.session import Session, MultiSession
from src.modules.CopyNTM import CopyNTM
from tests.minigrid.utils import lowpriority
from src.ga import GA


@click.command()
@click.option('--config_name', default="config_ntm_copy_2", help="Name of the config file")
@click.option('--config_folder', default="config_files/", help="folder where the config file is located")
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
        with open(Path(session.save_folder) / "config", "a") as fp:
            config.write(fp)
    session.start()


@click.command()
def plot():
    while True:
        res = da.get_results_from_session()
        df = da.results_to_dataframe(res)
        print(df)
        sns.lineplot(x='generation', y='max_score', hue='run', data=df,
                     err_style="bars", ci='run')
        plt.legend()
        plt.show()


@click.group()
def main():
    pass


main.add_command(run)
main.add_command(plot)

if __name__ == '__main__':
    main()
