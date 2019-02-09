"""
An example of how to use the choice environment
"""

import gym
import custom_envs

env_name = "MiniGrid-Choice3x1-color0-v0"  # goal is green
# env_name = "MiniGrid-Choice3x1-color1-v0"  # goal is blue
fps = 8

env = gym.make(env_name)

# # alternatively envs can be created manually with custom size, random starting position, and a custom number of
# # maximum steps before done (otherwise max_step = 2 * (width + height).
# from custom_envs.envs.choice import ChoiceEnv
# env = ChoiceEnv(1, width=3, height=3, random_positions=True, max_steps=10)


# show env
env.reset()
import time
env.render()
while True:
    time.sleep(1.0 / fps)
    step = observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if reward > 0:
        print("GOAL with reward: " + str(reward))
        time.sleep(3)
        break

