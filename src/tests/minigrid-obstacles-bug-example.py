
import random
import time
import gym
from gym_minigrid import *

if __name__ == '__main__':
    env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0")
    i = 0
    while True:
        action = random.randint(0, 6)
        observation, reward, done, info = env.step(action)
        # env.render("human")
        # time.sleep(1/20)
        i += 1
        if done:
            print(i)
            env.reset()