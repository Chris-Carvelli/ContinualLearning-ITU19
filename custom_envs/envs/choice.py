from gym_minigrid.minigrid import *

class ChoiceEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, goal_pos, width=3, height=1):
        self.goal_pos = goal_pos % 2
        assert width >= 3
        assert height >= 1
        super().__init__(
            width=width + 2,
            height=height + 2,
            max_steps=2 * (width + height),
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (int(width/2), height - 2)
        self.start_dir = 0

        i_left = self.goal_pos
        i_right = (self.goal_pos + 1) % 2

        left = Goal()
        left.color = IDX_TO_COLOR[1 + i_left]  # color
        left.type = IDX_TO_OBJECT[8 - i_left]  # Box or Goal

        right = Goal()
        right.color = IDX_TO_COLOR[1 + i_right]  # color
        right.type = IDX_TO_OBJECT[8 - i_right]  # Box or Goal

        # Place a goal square in the bottom-right corner
        self.grid.set(1, 1, left)
        self.grid.set(width - 2, 1, right)
        self.mission = "Get to the goal"
        self.goal_id = self.goal_pos

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1 - 0.9 * min(1, (self.step_count / self.max_steps))


class ChoiceEnv3x1_0(ChoiceEnv):
    def __init__(self):
        super().__init__(0, 3, 1)


class ChoiceEnv3x1_1(ChoiceEnv):
    def __init__(self):
        super().__init__(1, 3, 1)
