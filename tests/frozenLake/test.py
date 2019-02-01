import click
import gym


@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
def run(n_generations, n_processes):
    env = gym.make('FrozenLake-v0')
    print(env.action_space)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
