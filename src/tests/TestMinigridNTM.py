import gym
from gym_minigrid import *

from src.modules.NTM_MinigridModule import MinigridNTM


if __name__ == '__main__':
    env = gym.make("MiniGrid-SimpleCrossingS9N3-v0")
    for _ in range(5):
        ntm = MinigridNTM(10, 10)
        for _ in range(20):
            # x = np.random.randint(0, 2, (4,))
            # x = np.random.randn(4)*0.05 + 0.5
            ntm.start_history()
            print(ntm.evaluate(env, 50, render=True, fps=30))
            ntm.evolve(0.5)
            # ntm.plot_history()
            # print(x, ntm(x))
        print("-----------")

