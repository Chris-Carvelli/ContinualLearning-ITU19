import click
# import
from tests.frozenLake_chris.ga import GA


@click.command()
@click.option("--env_key", default='MiniGrid-Empty-6x6-v0')
@click.option("--n_generations", type=int, default=20)
@click.option("--pop_size", type=int, default=20)
@click.option("--sigma", type=float, default=0.005)
@click.option("--trunc", type=int, default=5)
@click.option("--render", is_flag=True)
def run(env_key, n_generations, pop_size, sigma, trunc, render):
    ga = GA(pop_size, env_key)
    ga.optimize(n_generations, sigma, trunc)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

