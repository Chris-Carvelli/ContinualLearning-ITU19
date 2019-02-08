import gym
import custom_envs

# env_name = "MiniGrid-Choice3x1-pos0-v0"
env_name = "MiniGrid-Choice3x1-pos1-v0"
fps = 5

env = gym.make(env_name)

# show env
env.reset()
import time
for _ in range(10000000):
    env.render()
    step = observation, reward, done, info = env.step(env.action_space.sample())
    time.sleep(1.0 / fps)
