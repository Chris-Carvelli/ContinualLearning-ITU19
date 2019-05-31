import random
import sys

import numpy
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import click

from data_analyzer import *
from configparser import ConfigParser
from pathlib import Path

from data_analyzer import SessionResult
from sessions.session import Session, MultiSession
from src.modules.CopyNTM import CopyNTM
from src.modules.NTM_TMazeModule import TMazeNTMModule
from src.utils import lowpriority, load
from src.ga import GA


@click.command()
@click.option('--config_name', default="config_ntm_copy_2", help="Name of the config file")
@click.option('--config_folder', default="config_files/copy", help="folder where the config file is located")
@click.option('--session_name', default=None, type=str, help='Session name. Default/None means same as config_file')
@click.option('--multi_session', default=1, help='Repeat experiment as a multi-session n times if n > 1')
@click.option('--pe', is_flag=True, help='Parallel (non-concurrent) execution for multi-session')
@click.option('--on_load', default=None,
              help='Specify a package and method for a session on_load method. For instance: src.utils:restart_session_after_errors')
def run(config_name, config_folder, session_name, multi_session, pe, on_load):
    lowpriority()
    if session_name is None:
        session_name = config_name if multi_session <= 1 else f"{config_name}-x{multi_session}"
        session_name = session_name.replace(".ini", "")

    # Load config
    config = ConfigParser()
    read_ok = config.read(f"{config_folder}/{config_name}")
    assert len(read_ok) > 0, f"Failed to read config file: {Path(config_folder) / config_name}"

    # Define modules here. NOTE no longer need. Module and path can be specified directly in config file now
    if "Controller" in config and "module" in config["Controller"]:
        module = config["Controller"]["module"]
        if module == "CopyNTM":
            model_builder = lambda: CopyNTM(**dict([(key, int(value)) for key, value in config["NTM"].items()]))
        elif module == "TMazeNTMModule":
            model_builder = lambda: TMazeNTMModule(**dict([(k, int(v)) for k, v in config["ModelParameters"].items()]))
        else:
            raise AssertionError(f"Unknown module specification: {module}")
    else:
        model_builder = None

    # Set seed
    seed = config["HyperParameters"].get("seed") if "HyperParameters" in config else None
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
        session = MultiSession(workers, session_name, parallel_execution=pe)
    else:
        session = Session(workers[0], session_name)

    # Copy config file to session folder
    if not os.path.isfile(Path(session.save_folder) / "config"):
        with open(Path(session.save_folder) / "config", "a") as fp:
            config.write(fp)
    if on_load:
        on_load = load(on_load)
    session.start(on_load=on_load)


@click.command()
@click.option('--ppo_results', default='', help='Path to .csv file containing results of PPO')
@click.option('--sessions_folder', default='Experiments', help='Path to folder containing session results')
@click.option('--sessions_to_load', default='',
              help='name of session folder(s) (.ses); if you want multiple, separate with comma')
@click.option('--hide_indv', is_flag=True, help="hides individual max, mean, median plots")
@click.option('--hide_merged', is_flag=True, help="hides individual averaged result of all runs for multi session")
@click.option('--use_explorer', is_flag=True, prompt='Use explorer?')
@click.option('--fill_missing_data', is_flag=True,
              help="For multisession sessions with less results will be copy data from last generation until they match")
@click.option('--max_gen', default="-1", help="Only data before max generation will be used")
@click.option('--save_name', default="", help="Name of figures to be saved")
@click.option('--plot_title', default="", help="Title for max/median/mean plots")
@click.option('--labels', default="", help="comma seperated list of labels for max/median/mean plots")
@click.option('--ymax', default="1.05", help="maximum y-value when plotting")
@click.option('--plot_elite_max', is_flag=True, help="plot max of elites as well as full generation")
def plot(ppo_results, sessions_folder, sessions_to_load, hide_indv, hide_merged, use_explorer, fill_missing_data,
         max_gen, save_name, plot_title, labels, ymax, plot_elite_max):
    # Get all sessions needed
    session_names = sessions_to_load.replace(" ", "").split(",")
    # label_dict = defaultdict(str)
    labels = dict([(k, v) for k, v in enumerate(labels.strip(" ").split(",") if labels else [])])
    # labels = labels.strip(" ").split(",") if labels else [None] * (len(session_names) + 1)
    results_objects = []
    max_gen = int(max_gen)
    ymax = float(ymax) if ymax else None
    # Get path to session
    # If not provided as argument propmt the user
    if not sessions_folder or not sessions_to_load:
        # True -> Use explorer
        # False -> Use terminal
        res_path = get_path_to_session(use_explorer)
        results_objects.append(SessionResult(_path=res_path, fill_missing_data=fill_missing_data))

        session_names = [Path(res_path).name]
        print(f"Plotting: {session_names[0]}")

    else:
        for name in session_names:
            results_objects.append(
                SessionResult(_path=Path(sessions_folder) / name, _name=name, fill_missing_data=fill_missing_data))

    # Get session and put the data in data frame
    # result_session = load_session(res_path)
    # df, is_single = results_to_dataframe(result_session)

    # Plot runs against each other if it's a multi-session
    for result, name in zip(results_objects, session_names):
        # y = "elite_max" if "elite_max" in result.split_df else "max_score"
        # for y in ["elite_max", "max_score"]:
        if not result.is_single:
            line = sns.lineplot(x="generation", y="max_score", hue="run",  data=result.split_df)
            line.set_title(f"Max Scores : {name}")
            plt.ylim(None, ymax)
        save_plot(Path(sys.argv[0]).parent / "plots", name)
        plt.show()
        if plot_elite_max:
            if not result.is_single:
                line = sns.lineplot(x="generation", y="elite_max", hue="run",  data=result.split_df)
                line.set_title(f"Max Scores of Elites: {name}")
                plt.ylim(None, ymax)
            save_plot(Path(sys.argv[0]).parent / "plots", f"{name}-elites")
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
        for result in results_objects:
            joined_data = [result.df['mean_score'], result.df['max_score'], result.df['median_score']]
            joined_data_plot = sns.lineplot(data=joined_data).set_title(result.name)
            plt.xlabel('Generations')
            plt.ylabel('Score')
            plt.legend(['mean', 'max', 'median'])
        plt.show()

    if not hide_indv:
        # for y in ["max_score", "mean_score", "median_score"]:
        y_names = ["elite_max", "max_score"] if plot_elite_max else ["max_score"]
        for y in y_names:
            if y not in results_objects[0].split_df:
                continue
            for i, result in enumerate(results_objects):
                data = result.split_df.loc[result.split_df["generation"] < max_gen] if max_gen > 0 else result.split_df
                sns.lineplot(x="generation", y=y, data=data, label=labels.get(i) or result.name).set_title(
                    plot_title or y)
            plt.ylim(None, ymax)
            if save_name:
                save_plot(Path(sys.argv[0]).parent / "plots", f"{save_name}-{y}")
            plt.show()

    if not sessions_to_load:
        main()



def save_plot(folder, name):
    os.makedirs(folder, exist_ok=True)
    name = name + ".png" if not name.endswith(".png") else name
    name = name.replace(".", ",").replace(",png", ".png")
    plt.savefig(folder / name, bbox_inches="tight")
    print(f"saved figure: {folder / name}")


@click.command()
@click.option("--fps", default="20", help="frames per second")
@click.option("--env_key", default="", help="frames per second")
@click.option("--max_eval", default="100", help='max number of evaluations')
def render(fps, env_key, max_eval):
    fps = int(fps)
    max_eval = int(max_eval)
    res_path = get_path_to_session(True)
    session = load_session(res_path)
    if isinstance(session, MultiSession):
        # worker = session.workers[0]
        worker = sorted([(w, w.results[-1]["scored_parents"][0][1]) for w in session.workers], key=lambda x: x[1])[-1][0]
    else:
        worker = session.worker
    if env_key:
        import gym
        envs = [gym.make(env_key)]
    else:
        envs = worker.envs if hasattr(worker, "envs") else [worker.env]
    champ, max_score = worker.tuple_results()[-1][-1][0]
    print(f"Champ has max-score = {max_score}")
    while True:
        for i, env in enumerate(envs):
            res = champ.evaluate(env, int(max_eval), render=True, fps=int(fps))
            print(f"env({i}) - Evaluates to {res}")


@click.group()
def main():
    pass


main.add_command(run)
main.add_command(plot)
main.add_command(render)

if __name__ == '__main__':
    main()
