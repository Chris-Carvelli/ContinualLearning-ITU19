import os
import random
import click
import numpy
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from data_analyzer import *
from custom_envs import *
from configparser import ConfigParser
from pathlib import Path
from sessions.session import Session, MultiSession
from src.modules.CopyNTM import CopyNTM
from src.modules.NTM_TMazeModule import TMazeNTMModule
from tests.minigrid.utils import lowpriority
from src.ga import GA


@click.command()
@click.option('--config_name', default="config_ntm_copy_2", help="Name of the config file")
@click.option('--config_folder', default="config_files/copy", help="folder where the config file is located")
@click.option('--session_name', default=None, type=str, help='Session name. Default/None means same as config_file')
@click.option('--multi_session', default=1, help='Repeat experiment as a multi-session n times if n > 1')
def run(config_name, config_folder, session_name, multi_session):
    lowpriority()
    if session_name is None:
        session_name = config_name if multi_session <= 1 else f"{config_name}-x{multi_session}"

    # Load config
    config = ConfigParser()
    read_ok = config.read(f"{config_folder}/{config_name}")
    assert len(read_ok) > 0, f"Failed to read config file: {Path(config_folder) / config_name}"

    def config_get(section, key, default=None):
        return config[section][key] if section in config and key in config[section] else default

    # Define modules here
    module = config_get("Controller", "module")
    if module == "CopyNTM":
        model_builder = lambda: CopyNTM(**dict([(key, int(value)) for key, value in config["NTM"].items()]))
    elif module == "TMazeNTMModule":
        model_builder = lambda: TMazeNTMModule(**dict([(k, int(v)) for k, v in config["ModelParameters"].items()]))
    else:
        raise AssertionError(f"Unknown module specification: {module}")

    # Set seed
    seed = int(config_get("HyperParameters", "seed"))
    if seed:
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

    # Create worker(s) and session
    workers = []
    for i in range(max(1, multi_session)):
        workers.append(GA(f"{config_folder}/{config_name}", model_builder=model_builder))
    if multi_session > 1:
        session = MultiSession(workers, session_name)
    else:
        session = Session(workers[0], session_name)

    # Copy config file to session folder
    if not os.path.isfile(Path(session.save_folder) / "config"):
        with open(Path(session.save_folder) / "config", "a") as fp:
            config.write(fp)

    session.start()


@click.command()
@click.option('--ppo_results', default='', help='Path to .csv file containing results of PPO')
@click.option('--session_results', default='', help='Path to .ses folder to analyze')
def plot(ppo_results, session_results):
    # Get path to session
    # If not provided as argument propmt the user
    if not session_results:
        # True -> Use explorer
        # False -> Use terminal
        res_path = get_path_to_session(False)
    else:
        res_path = session_results

    # Get session and put the data in data frame
    result_session = load_session(res_path)
    df, is_single = results_to_dataframe(result_session)

    # Plot runs against each other if it's a multi-session
    if not is_single:
        sns.lineplot(x="generation", y="max_score", hue="run", data=df).set_title("Max Score per run")
        plt.show()

    # ppo_results = "C:\\Users\\Luka\\Documents\\Python\\minigrid_rl\\torch-rl\\storage\\DoorKey"
    # Get results as dataframe from minigrid_rl. You need to provide the path of the log
    if ppo_results:
        df2 = get_ppo_results(ppo_results)
        # print(df2.columns.values)
        print(df.columns.values)
        print(df2.columns.values)
        # print(df2)
        print(df2)
        # df = df.set_index('generation').join(df2.set_index('update'))

    # Combine all runs into one and do an average of the results if is multisession
    if not is_single:
        df = df.groupby('generation').mean()
    # Plot combined data
    joined_data = [df['mean_score'], df['max_score'], df['median_score']]
    joined_data_plot = sns.lineplot(data=joined_data).set_title("Max, Mean and Median Score Averaged over all runs")
    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.legend(['mean', 'max', 'median'])

    plt.show()


@click.group()
def main():
    pass


main.add_command(run)
main.add_command(plot)

if __name__ == '__main__':
    main()
