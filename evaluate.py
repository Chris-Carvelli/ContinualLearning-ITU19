import click
from pathlib import Path
from typing import List

import gym
from custom_envs import *

from data_analyzer import *
from src.ga import GA
from src.modules.NTM_MinigridModule import MinigridNTM
from src.modules.NTM_TMazeModule import TMazeNTMModule
from src.modules.CopyNTM import CopyNTM


@click.command()
@click.option("--max_eval", default="100000", help='max number of evaluations')
@click.option("--render/--no-render", default=False, help="rendeing or no rendering")
@click.option("--fps", default="10", help="frames per second")
@click.option('--use_explorer', is_flag=True, prompt='Use explorer?')
@click.option('--plot_data', is_flag=True, prompt='Plot evolution data?')
def evaluate(max_eval, render, fps, use_explorer, plot_data):
    res_path = get_path_to_session(use_explorer)
    print(f"Evaluating: {Path(res_path).name}")
    result = SessionResult(_path=res_path)
    if plot_data:
        sns.lineplot(x="generation", y="max_score", hue="run", data=result.split_df).set_title(
            f"Max Scores : {Path(res_path).name}")
        plt.show()
    max_eval = None if max_eval == "" else int(max_eval)
    fps = int(fps)
    workers: List[GA] = result.session.workers
    for i, worker in enumerate(workers):
        champ, max_score = worker.tuple_results()[-1][-1][0]
        if isinstance(champ, CopyNTM):
            eval_score = evaluate_copy_task(champ, max_eval=max_eval, render=render, fps=fps)
        elif isinstance(champ, TMazeNTMModule):
            eval_score = evaluate_t_maze_task(worker, champ, max_eval=max_eval, render=render, fps=fps)
        elif isinstance(champ, MinigridNTM):
            eval_score = evaluate_minigrid_task(champ, worker.envs, max_eval=max_eval, render=render, fps=fps)
        else:
            raise NotImplementedError()
        print(f"worker({i}) scored {eval_score:.4f} (got {max_score:.4f} during evolution)")


def evaluate_minigrid_task(nn: CopyNTM, envs, max_eval, render, fps):
    rewards = []
    for env in envs:
        if isinstance(env, gym.Wrapper):
            env = env.unwrapped
        env_rewards = []
        for _ in range(50):
            reward, n_eval = nn.evaluate(env, max_eval, render=False)
            env_rewards.append(reward)
        if hasattr(nn, "history"):
            nn.start_history()
        reward, n_eval = nn.evaluate(env, max_eval, render=render, fps=fps)
        env_rewards.append(reward)
        if hasattr(nn, "history"):
            nn.plot_history()
        rewards += [sum(env_rewards) / len(env_rewards)]
    if len(envs) > 1:
        print(f"Individual envs got rewards: {rewards}")
    return sum(rewards) / len(rewards)


def evaluate_copy_task(nn: CopyNTM, max_eval, render, fps):
    h = nn.memory_unit_size - 2
    rewards = []
    for l in [5, 10, 20, 30, 50, 100]:
        env = gym.make(f"Copy-{h}x{l}-v0")
        for _ in range(5):
            tot_reward, n_eval = nn.evaluate(env, max_eval, render=False)
            rewards.append(tot_reward)
    nn.start_history()
    env = gym.make(f"Copy-{h}x{4}-v0")
    tot_reward, n_eval = nn.evaluate(env, max_eval, render=False, fps=fps)
    rewards.append(tot_reward)
    nn.plot_history(vmin=0, vmax=1)
    return sum(rewards) / len(rewards)


def evaluate_t_maze_task(worker: GA, nn: TMazeNTMModule, max_eval, render, fps):
    from custom_envs.envs import TMaze
    base_env: TMaze = worker.env
    view_size = base_env.view_size
    double = len(base_env.goal_positions) == 4
    length = base_env.env.corridor_length
    rewards = []
    for rounds in [10]:
        env = TMaze(length, rounds, cyclic_order=True, view_size=view_size, double=double, uneven_rounds=True, repeat=2)
        for _ in range(len(env.permutations)):
            reward, n_eval = nn.evaluate(env, max_eval, render=False)
            rewards.append(reward)
    env = TMaze(length, 10, cyclic_order=False, view_size=view_size, double=double, uneven_rounds=True,
                goal_positions=[1, 0, 1, 0])
    for _ in range(100):
        reward, n_eval = nn.evaluate(env, max_eval, render=False)
        rewards.append(reward)
    nn.start_history()
    env = TMaze(length, 3, cyclic_order=False, view_size=view_size, double=double, uneven_rounds=True, repeat=2)
    reward, n_eval = nn.evaluate(env, max_eval, render=render, fps=fps)
    rewards.append(reward)
    nn.plot_history(vmin=0, vmax=1)
    return sum(rewards) / len(rewards)


if __name__ == '__main__':
    evaluate()
