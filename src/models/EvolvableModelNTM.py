

class EvolvableModel:
    def __init__(self, ntm):
        self.ntm = ntm
        self.ntm.init()

    def evolve(self, sigma):
        self.ntm.evolve(sigma)

    def evaluate(self, env, max_eval, render=False, fps=60):
        return self.ntm.evaluate(
            env=env,
            max_eval=max_eval,
            render=render,
            fps=fps
        )
