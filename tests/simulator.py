import sys
import time
import torch
import numpy as np

from torch.autograd import Variable


def simulate(env, model=None, fps=5, env_type="minigrid"):
    if env_type == "minigrid":
        state = env.reset()
        env.render()
        sys.stdout.write('Rewards:')
        while True:
            state = state['image']
            if model is not None:
                values = model(Variable(torch.Tensor([state])))
                action = np.argmax(values.data.numpy()[:env.action_space.n])
            else:
                action = env.action_space.sample()
            step = state, reward, done, info = env.step(action)
            time.sleep(1/fps)
            env.render()
            sys.stdout.write(f"{reward} ")
            if done:
                print("\nGOAL with reward: " + str(reward))
                time.sleep(1/fps)
                break
    else:
        raise NotImplementedError()