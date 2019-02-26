class EvolvableModel:
    def __init__(self, pnn):
        self.pnn = pnn
        self.pnn.init()

    def evolve(self, sigma):
        self.pnn.evolve(sigma)

    def evaluate(self, env, max_eval, render=False, fps=60):
        return self.pnn.evaluate(
            env=env,
            max_eval=max_eval,
            render=render,
            fps=fps
        )

