import os
import random
import re

import numpy
import torch
import pandas as pd

from data_analyzer import *
from custom_envs import *
from configparser import ConfigParser
from pathlib import Path
from sessions.session import Session, MultiSession, MultiThreadedSession
from src.modules.CopyNTM import CopyNTM
from src.modules.NTM_Module import NTM
from src.modules.NTM_TMazeModule import TMazeNTMModule
from src.utils import lowpriority, int_or_float
from src.ga import GA


class SessionResult:
    def __init__(self, _path, _df=None, _name="NoName", _is_single=True):
        self.split_df = None
        self.session_path = _path
        self.is_single = _is_single
        if _df is None:
            self.load_data()
        else:
            self.df = _df
        self.name = _name

    def load_data(self):
        df, is_single = results_to_dataframe(load_session(self.session_path))
        self.df = df
        self.is_single = is_single
        if not self.is_single:
            self.split_df = self.df
            self.df = self.df.groupby('generation').mean()

@click.command()
@click.option('--config_name', default="config_ntm_copy_2", help="Name of the config file")
@click.option('--config_folder', default="config_files/copy", help="folder where the config file is located")
@click.option('--session_name', default=None, type=str, help='Session name. Default/None means same as config_file')
@click.option('--multi_session', default=1, help='Repeat experiment as a multi-session n times if n > 1')
@click.option('--mt', is_flag=True, help='use multiple thread for multi-session')
@click.option('--pe', is_flag=True, help='Parallel (sequential) execution for multi-session')
def run(config_name, config_folder, session_name, multi_session, mt, pe):
    lowpriority()
    if session_name is None:
        session_name = config_name if multi_session <= 1 else f"{config_name}-x{multi_session}"
        session_name = session_name.replace(".ini", "")

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
    seed = config_get("HyperParameters", "seed")
    if seed:
        seed = int(seed)
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

    # Create worker(s) and session
    workers = []
    for i in range(max(1, multi_session)):
        workers.append(GA(f"{config_folder}/{config_name}", model_builder=model_builder))
    if multi_session > 1:
        if mt:
            raise NotImplementedError("MultiThreadedSessin not yet implemented")
            # session = MultiThreadedSession(workers, session_name)
        else:
            session = MultiSession(workers, session_name, parallel_execution=pe)
    else:
        session = Session(workers[0], session_name)

    # Copy config file to session folder
    if not os.path.isfile(Path(session.save_folder) / "config"):
        with open(Path(session.save_folder) / "config", "a") as fp:
            config.write(fp)

    session.start()


@click.command()
@click.option('--ppo_results', default='', help='Path to .csv file containing results of PPO')
@click.option('--sessions_folder', default='', help='Path to folder containing session results')
@click.option('--sessions_to_load', default='',
              help='name of session folder(s) (.ses); if you want multiple, separate with comma')
@click.option('--hide_indv', is_flag=True, help="hides individual max, mean, median plots")
@click.option('--hide_merged', is_flag=True, help="hides individual averaged result of all runs for multi session")
@click.option('--hide_merged', is_flag=True, help="hides individual averaged result of all runs for multi session")
@click.option('--use_explorer', is_flag=True, prompt='Use explorer?')
def plot(ppo_results, sessions_folder, sessions_to_load, hide_indv, hide_merged, use_explorer):
        # Get all sessions needed
        session_names = sessions_to_load.replace(" ", "")
        session_names = session_names.split(",")
        results_objects = []
        # Get path to session
        # If not provided as argument propmt the user
        if not sessions_folder or not sessions_to_load:
            # True -> Use explorer
            # False -> Use terminal
            res_path = get_path_to_session(use_explorer)
            results_objects.append(SessionResult(_path=res_path))

            session_names = [Path(res_path).name]
            print(f"Plotting: {session_names[0]}")

        else:
            for name in session_names:
                results_objects.append(SessionResult(_path=sessions_folder + name, _name=name))

        # Get session and put the data in data frame
        # result_session = load_session(res_path)
        # df, is_single = results_to_dataframe(result_session)

        # Plot runs against each other if it's a multi-session
        for result, name in zip(results_objects, session_names):
            if not result.is_single:
                sns.lineplot(x="generation", y="max_score", hue="run", data=result.split_df).set_title(
                    f"Max Scores : {name}")
                plt.show()

        # TODO: Make data comparable to PPO
        # ppo_results = "C:\\Users\\Luka\\Documents\\Python\\minigrid_rl\\torch-rl\\storage\\DoorKey"
        # Get results as dataframe from minigrid_rl. You need to provide the path of the log
        if ppo_results:
            df2 = get_ppo_results(ppo_results)
            # print(df2.columns.values)
            #    print(df.columns.values)
            print(df2.columns.values)
            # print(df2)
            print(df2)
            # df = df.set_index('generation').join(df2.set_index('update'))

        if not hide_merged:
            # Combine all runs into one and do an average of the results if is multisession
            # if not is_single:
            #    df = df.groupby('generation').mean()
            for result in results_objects:
                joined_data = [result.df['mean_score'], result.df['max_score'], result.df['median_score']]
                joined_data_plot = sns.lineplot(data=joined_data).set_title(result.name)
                plt.xlabel('Generations')
                plt.ylabel('Score')
                plt.legend(['mean', 'max', 'median'])
                plt.show()

        if not hide_indv:
            mean_df = pd.DataFrame()
            max_df = pd.DataFrame()
            median_df = pd.DataFrame()

            for result in results_objects:
                mean_df[result.name] = result.df['mean_score']
                max_df[result.name] = result.df['max_score']
                median_df[result.name] = result.df['median_score']
            sns.lineplot(data=mean_df).set_title("Mean")
            plt.show()
            sns.lineplot(data=max_df).set_title("Max")
            plt.show()
            sns.lineplot(data=median_df).set_title("Median")
            plt.show()

        if not (sessions_folder or sessions_to_load):
            main()


@click.command()
@click.option("--max_eval", default="100", help='max number of evaluations')
@click.option("--render/--no-render", default=True, help="rendeing or no rendering")
@click.option("--fps", default="60", help="frames per second")
def evaluate(max_eval, render, fps):
    max_eval = int(max_eval)
    fps = int(fps)
    while True:
        res_path = get_path_to_session(False)
        session = load_session(res_path)
        if isinstance(session, MultiSession):
            worker = session.workers[3]
        else:
            worker = session.worker
        if isinstance(worker, GA):
            env = worker.env
            import gym
            env = gym.make(f"TMaze-{4}x{5}-viewsize_{3}-v0")
            nn, max_score = worker.results[-1][-1][0]
            if isinstance(nn, NTM):
                nn.start_history()
            tot_reward, n_eval = nn.evaluate(env, int(max_eval), render=render, fps=int(fps))
            print(f"Evaluates to reward: {tot_reward}")
            if isinstance(nn, NTM):
                nn.plot_history()


@click.group()
def main():
    pass


main.add_command(run)
main.add_command(plot)
main.add_command(evaluate)

if __name__ == '__main__':
    main()
