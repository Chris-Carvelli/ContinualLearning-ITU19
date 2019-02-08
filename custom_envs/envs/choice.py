
from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class ChoiceEnv3x1(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, goal_pos):
        self.goal_pos = goal_pos % 2
        super().__init__(
            width=3 + 2,
            height=1 + 2,
            max_steps=2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (2, 1)
        self.start_dir = 0

        i_left = self.goal_pos
        i_right = (self.goal_pos + 1) % 2

        left = Goal()
        left.color = IDX_TO_COLOR[1 + i_left]  # color
        left.type = IDX_TO_OBJECT[7 + i_left]  # Box or Goal

        right = Goal()
        right.color = IDX_TO_COLOR[1 + i_right]  # color
        right.type = IDX_TO_OBJECT[7 + i_right]  # Box or Goal

        # Place a goal square in the bottom-right corner
        self.grid.set(1, 1, left)
        self.grid.set(3, 1, right)
        self.mission = "Get to the goal"
        self.goal_id = self.goal_pos


class ChoiceEnv3x1_0(ChoiceEnv3x1):
    def __init__(self):
        super().__init__(0)


class ChoiceEnv3x1_1(ChoiceEnv3x1):
    def __init__(self):
        super().__init__(1)

# i need this https://github.com/MartinThoma/banana-gym/tree/master/gym_banana

