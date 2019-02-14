from gym_minigrid.minigrid import *
import random
from custom_envs.maze import Maze


def dist(pos1, pos2):
    """Returns distance between two points"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


class ChoiceEnv(MiniGridEnv):
    """
    An empty grid environment, with a blue square (top, left) and a green square (top, right).
    The goal is either to reach blue or the green square.
    """

    def __init__(self, goal_color=0, width=3, height=1, random_positions=False, max_steps=None, maze_env=False,
                 see_through_walls=True, euclid_dist_reward=False):
        """
        :param goal_color: Determines if the goal is set to the blue or green square. Allowed values are 0 and 1
        :param width: The width of the env
        :param height: The height of the env
        :param random_positions: if true the squares will have random starting positions
        :param max_steps: The maximum number of steps that can be taken before the env return done=True
        """
        self.euclid_dist_reward = euclid_dist_reward
        self.goal_color = goal_color
        self.maze_env = maze_env
        self.random_positions = random_positions
        assert 0 <= goal_color <= 2
        assert width >= 1
        assert height >= 1
        assert width * height >= 3
        if maze_env:
            assert width >= 3 and height >= 3, "Maze env must be at least 3x3"
            assert width % 2 == 1, "width of a maze env must be uneven"
            assert height % 2 == 1, "height of a maze env must be uneven"
            assert random_positions, "maze_env must have random positions"
        if max_steps is None:
            max_steps = 2 * (width + height)
        super().__init__(
            width=width + 2,
            height=height + 2,
            max_steps=max_steps,
            see_through_walls=see_through_walls  # Set this to True for maximum speed
        )
        self.start_dist = dist(self.goal.cur_pos, self.start_pos)

    def step(self, action):
        pos_before = self.agent_pos
        obs, reward, done, _ = super().step(action)
        if self.euclid_dist_reward and action == self.actions.forward and reward <= 0:
            dist0 = dist(pos_before, self.goal.cur_pos)
            dist1 = dist(self.agent_pos, self.goal.cur_pos)
            reward += (dist0 - dist1)/self.start_dist
        return obs, reward, done, _

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (int(width / 2), height - 2)
        self.start_dir = 3

        if self.maze_env:
            self.start_pos = (1, height - 2)
            m = Maze(int((self.width - 1) / 2), int((self.height - 1) / 2))
            m.randomize()
            print(m.__repr__())
            m = m._to_str_matrix()
            for y, row in enumerate(m):
                for x, cell in enumerate(row):
                    if cell == "O" and 0 <= x < self.width and 0 <= y < self.height:
                        self.grid.set(x, y, Wall())

        color0 = Goal()
        color0.color = IDX_TO_COLOR[1]  # color
        color0.type = IDX_TO_OBJECT[8 - self.goal_color]  # Box or Goal

        color1 = Goal()
        color1.color = IDX_TO_COLOR[2]  # color
        color1.type = IDX_TO_OBJECT[8 - (self.goal_color + 1) % 2]  # Box or Goal

        if self.random_positions:
            pos = self.get_random_free_position()
            self.grid.set(pos[0], pos[1], color0)
            pos = self.get_random_free_position()
            self.grid.set(pos[0], pos[1], color1)
            color1.cur_pos = pos
        else:
            self.grid.set(1, 1, color0)
            color0.cur_pos = (1, 1)
            self.grid.set(width - 2, 1, color1)
            color1.cur_pos = (width - 2, 1)
        self.mission = str(self.goal_color)
        if self.goal_color == 0:
            self.goal = color0
        else:
            self.goal = color1
        # self.goal_pos =
        # self.mission = "Get to the " + IDX_TO_COLOR[1 + self.goal_pos] + " square"

    def get_random_free_position(self):
        """Returns the coordinates as an integer tuple, of a free tile"""
        pos = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))
        while pos == self.start_pos or self.grid.get(pos[0], pos[1]) is not None:
            pos = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))
        return pos

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1 - 0.9 * min(1, (self.step_count / self.max_steps))


class ChoiceEnv3x1_0(ChoiceEnv):
    def __init__(self):
        super().__init__(0, 3, 1, max_steps=20)


class ChoiceEnv3x1_1(ChoiceEnv):
    def __init__(self):
        super().__init__(1, 3, 1, max_steps=20)
