from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from random import random


class EmptyEnvNoise(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1 + int(random() * (width / 2)), 1 + int(random() * (height / 2)))
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        self.mission = "get to the green goal square"


class EmptyEnvNoise6x6(EmptyEnvNoise):
    def __init__(self):
        super().__init__(size=6)


class EmptyEnvNoise16x16(EmptyEnvNoise):
    def __init__(self):
        super().__init__(size=16)
