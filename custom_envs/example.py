import gym
import custom_envs

# env_name = "MiniGrid-Choice3x1-pos0-v0"  # goal is at the left
env_name = "MiniGrid-Choice3x1-pos1-v0"  # goal is at the right
fps = 8

env = gym.make(env_name)

# # alternatively envs can be created manually with custom size like this
# from custom_envs.envs.choice import ChoiceEnv
# env = ChoiceEnv(0, 5, 5)

# show env
env.reset()
import time
while True:
    env.render()
    step = observation, reward, done, info = env.step(env.action_space.sample())
    time.sleep(1.0 / fps)
    if reward > 0:
        print("GOAL with reward: " + str(reward))
        time.sleep(3)
        break

