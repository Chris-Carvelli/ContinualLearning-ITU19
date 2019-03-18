import random
from typing import List, Tuple
from gym_minigrid.minigrid import *
from gym_minigrid.minigrid import Grid

from custom_envs.envs.multi_env import MultiEnv


class AbstractTMaze(MiniGridEnv):
    reward_position: int
    grid: Grid
    corridor_length: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _reward(self):
        min_steps = (1 + 2 * self.corridor_length)
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

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            else:
                reward = -1
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True


        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}


class SingleTMaze(AbstractTMaze):

    def __init__(self, corridor_length=3, reward_position=0, max_steps=None):
        self.reward_position = reward_position
        self.corridor_length = corridor_length
        assert corridor_length > 0

        if max_steps is None:
            max_steps = 2 * (corridor_length + 1) + 2

        if self.reward_position == 0:
            self.mission = "Go to reward to the LEFT"
        else:
            self.mission = "Go to reward to the RIGHT"
        super().__init__(
            width=3 + 2 * corridor_length,
            height=3 + corridor_length,
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

    def __init__(self, corridor_length=3, reward_position=0, max_steps=None):
        self.reward_position = reward_position
        self.corridor_length = corridor_length
        assert corridor_length > 0

        if max_steps is None:
            max_steps = 4 + 4 * corridor_length

        goals = ["UPPER LEFT", "UPPER RIGHT", "LOWER RIGHT", "LOWER LEFT", ]
        self.mission = f"Go to reward to the {goals[self.reward_position]}"

        super().__init__(
            width=3 + 2 * corridor_length,
            height=3 + 2 * corridor_length,
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
            for x in range(1, width - 1):
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

    def __init__(self, corridor_length=3, rounds_pr_side=10, max_steps=None, rnd_order=False):
        envs = [SingleTMaze(corridor_length, 0, max_steps),
                SingleTMaze(corridor_length, 1, max_steps)]
        self.rnd_order = rnd_order
        if self.rnd_order:
            random.shuffle(envs)
        super().__init__(envs, rounds_pr_side)
        self.total_rounds = self.total_rounds - 2

    def reset(self):
        if self.rnd_order:
            random.shuffle(self.schedule)
        return super().reset()

    def seed(self, seed=None):
        if self.rnd_order:
            random.seed(seed)
            if self.round == 0 and self.i == 0:
                random.shuffle(self.schedule)
        super().seed(seed)


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
    env.seed(1)
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
              ([2] * length + [1] + [2] * length) * (rounds - 1) \
              # + ([2] * length + [1] + [2] * length)
    env.render()
    total_reward = 0
    for a in actions:
        state, reward, done, info = env.step(a)
        time.sleep(.3)
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
