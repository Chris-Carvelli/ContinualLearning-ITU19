
import random
import time
import gym
from gym_minigrid import *
from custom_envs import *

if __name__ == '__main__':
    env = gym.make("TMaze-2x4-3-UnevenRounds-x2-v0")
    i = 0
    while True:
        action = random.randint(0, 6)
        env.render("human")
        observation, reward, done, info = env.step(4)
        env.render("human")
        time.sleep(100 + 1/20)
        i += 1
        if done:
            print(i)
            env.reset()