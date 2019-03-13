import torch

from functools import reduce

Z_VECT_EVOLUTION_PROBABILITY = 0.5


class Controller:
    def __init__(self, pnn, hnn):
        self.pnn = pnn
        self.hnn = hnn

        self.pnn.init()
        self.hnn.init()

        self._update_weights()

    def evolve(self, sigma):
        self.hnn.evolve(sigma)
        self._update_weights()

    def evaluate(self, env, max_eval, render=False, fps=60):
        return self.pnn.evaluate(
            env=env,
            max_eval=max_eval,
            render=render,
            fps=fps
        )

    def _update_weights(self):
        z_chunk = 0
        for name, param in self.pnn.named_parameters():
            if 'weight' in name:
                param.data = self._get_weights(z_chunk, param.shape).data
                z_chunk += 1

    def _get_weights(self, layer_index, layer_shape):
        w = self.hnn(layer_index)
        w = torch.narrow(w, 0, 0, reduce((lambda x, y: x * y), layer_shape))
        w = w.view(layer_shape)

        return w

