import click
import gym


@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
@click.option("--render", is_flag=True)
def run(n_generations, n_processes, render):
    env = gym.make('FrozenLake-v0')
    print(env.action_space)
    print(env.observation_space)

    obs = env.reset()

    tot_reward = 0
    done = False
    while not done:
        obs, reward, done, _ = env.step(env.action_space.sample())
        if render:
            env.render()

        tot_reward += reward
    print(tot_reward)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
