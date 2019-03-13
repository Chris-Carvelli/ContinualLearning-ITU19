from typing import List, Tuple
from gym_minigrid.minigrid import *
from gym_minigrid.minigrid import Grid

from custom_envs.envs.multi_env import MultiEnv


class AbstractTMaze(MiniGridEnv):
    reward_position: int
    grid: Grid

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _reward(self):
        min_steps = (1 + 2 * self.corrider_length)
        redundant_steps = max(0, self.step_count - min_steps)
        max_steps = self.max_steps - min_steps + 1
        max_reward = 0.1
        cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])
        if hasattr(cell, "is_goal") and cell.is_goal:
            max_reward = 0.9
        return max_reward * (1 - min(1, (redundant_steps / max_steps)))

    def _gen_rewards(self, rewards_pos: List[Tuple[int, int]]):
        for i, (x, y) in enumerate(rewards_pos):
            g = Goal()
            self.grid.set(x, y, g)
            g.is_goal = False
            if self.reward_position == i % len(rewards_pos):
                g.is_goal = True

    def close(self):
        if self.grid_render:
            self.grid_render.close()
            if self.grid_render.window:
                self.grid_render.window.close()


class SingleTMaze(AbstractTMaze):

    def __init__(self, corrider_length=3, reward_position=0, max_steps=None):
        self.reward_position = reward_position
        self.corrider_length = corrider_length
        assert corrider_length > 0

        if max_steps is None:
            max_steps = 4 + 4 * corrider_length

        if self.reward_position == 0:
            self.mission = "Go to reward to the LEFT"
        else:
            self.mission = "Go to reward to the RIGHT"
        super().__init__(
            width=3 + 2 * corrider_length,
            height=3 + corrider_length,
            max_steps=max_steps,
            see_through_walls=True  # True for maximum performance
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (int(width / 2), height - 2)
        self.start_dir = 3

        # Create walls
        for y in range(2, height - 1):
            for x in range(1, int(width / 2)):
                self.grid.set(x, y, Wall())
            for x in range(int(width / 2) + 1, width - 1):
                self.grid.set(x, y, Wall())

        # Create rewards
        reward_positions = [
            (1, 1),
            (width - 2, 1),
        ]
        self._gen_rewards(reward_positions)


class DoubleTMaze(AbstractTMaze):

    def __init__(self, corrider_length=3, reward_position=0, max_steps=None):
        self.reward_position = reward_position
        self.corrider_length = corrider_length
        assert corrider_length > 0

        if max_steps is None:
            max_steps = 4 + 4 * corrider_length

        goals = ["UPPER LEFT", "UPPER RIGHT", "LOWER RIGHT", "LOWER LEFT", ]
        self.mission = f"Go to reward to the {goals[self.reward_position]}"

        super().__init__(
            width=3 + 2 * corrider_length,
            height=3 + 2 * corrider_length,
            max_steps=max_steps,
            see_through_walls=True  # True for maximum performance
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (int(width / 2), int(height / 2))
        self.start_dir = 3

        # Create walls
        for y in range(2, height - 2):
            for x in range(1,  width - 1):
                if x == int(width / 2):
                    continue
                self.grid.set(x, y, Wall())

        # Create rewards
        reward_positions = [
            (1, 1),
            (width - 2, 1),
            (width - 2, height - 2),
            (1, height - 2),
        ]
        self._gen_rewards(reward_positions)


class TMaze(MultiEnv):

    def __init__(self, corrider_length=3, rounds_pr_side=10, max_steps=None):
        envs = [SingleTMaze(corrider_length, 0, max_steps),
                SingleTMaze(corrider_length, 1, max_steps)]
        super().__init__(envs, rounds_pr_side)
        self.total_rounds = self.total_rounds - 2


def test_one_shot_tmaze():
    import time
    length = 2
    # env = SingleTMaze(length, 0)
    env = DoubleTMaze(length, 0)

    # Actions
    # left = 0
    # right = 1
    # forward = 2
    # toggle = 5
    actions = [2] * length + [0] + [2] * length
    env.render()
    for a in actions:
        state, reward, done, info = env.step(a)
        time.sleep(.3)
        env.render()
        print(reward, done)
        if done:
            assert reward >= .9
    time.sleep(1)


def test_tmaze():
    import time
    rounds = 2
    length = 1
    env = TMaze(length, rounds)
    state = env.reset()
    del state["image"]
    print(state)

    # Actions
    # left = 0
    # right = 1
    # forward = 2
    # toggle = 5
    actions = [2] * length + [1] + [2] * length + \
              ([2] * length + [0] + [2] * length) * rounds + \
              ([2] * length + [1] + [2] * length) * (rounds - 1)
    env.render()
    total_reward = 0
    for a in actions:
        state, reward, done, info = env.step(a)
        time.sleep(.8)
        env.render()
        total_reward += reward
        del state["image"]
        print(reward, done, state)
        if done:
            assert total_reward >= 1
    time.sleep(1)


if __name__ == '__main__':
    # test_one_shot_tmaze()
    test_tmaze()

