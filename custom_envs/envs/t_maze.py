import random
import sys
import time
import gym
from typing import List, Tuple

from gym_minigrid.minigrid import Grid, MiniGridEnv, Wall, Goal
from custom_envs.envs.multi_env import MultiEnv


class SingleTMaze(MiniGridEnv):
    is_double = False
    reward_values = dict(goal=1, fake_goal=0.1)
    view_size: int = None

    def __init__(self, corridor_length=3, reward_position=0, max_steps=None, is_double=False, view_size=None,
                 max_corridor_length=None):
        if max_corridor_length is None:
            max_corridor_length = corridor_length
        self.max_corridor_length = max_corridor_length
        self.view_size = view_size if view_size is not None else 7
        self.is_double = is_double
        self.reward_position = reward_position
        self.corridor_length = corridor_length
        assert corridor_length > 0

        if max_steps is None:
            max_steps = 4 + 4 * corridor_length

        goals = ["UPPER LEFT", "UPPER RIGHT", "LOWER RIGHT", "LOWER LEFT", ]
        self.mission = f"Goal is {goals[self.reward_position]}"

        super().__init__(
            grid_size=3 + 2 * self.max_corridor_length,
            max_steps=max_steps,
            see_through_walls=True,  # True for maximum performance
            agent_view_size=self.view_size,
        )
        self.reward_range = (min(self.reward_values["fake_goal"], 0), self.reward_values["goal"])

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Place the agent in the top-left corner
        self.start_pos = (int(width / 2), int(height / 2))
        self.start_dir = 3

        # Create walls
        for x in range(0, width):
            for y in range(0, height):
                self.grid.set(x, y, Wall())

        # Create paths
        if self.is_double:
            for y in range(height // 2 - self.corridor_length, height // 2 + self.corridor_length + 1):
                self.grid.set(width // 2, y, None)
            for x in range(width // 2 - self.corridor_length, width // 2 + self.corridor_length + 1):
                self.grid.set(x, height // 2 - self.corridor_length, None)
                self.grid.set(x, height // 2 + self.corridor_length, None)
        else:
            for y in range(height // 2 - self.corridor_length, height // 2 + 1):
                self.grid.set(width // 2, y, None)
            for x in range(width // 2 - self.corridor_length, width // 2 + self.corridor_length + 1):
                self.grid.set(x, height // 2 - self.corridor_length, None)

        # Create rewards
        reward_positions = self._reward_positions(width, height)
        self._gen_rewards(reward_positions)

    def _reward_positions(self, width, height):
        reward_positions = [
            (width // 2 - self.corridor_length, height // 2 - self.corridor_length),
            (width // 2 + self.corridor_length, height // 2 - self.corridor_length),
            (width // 2 + self.corridor_length, height // 2 + self.corridor_length),
            (width // 2 - self.corridor_length, height // 2 + self.corridor_length),
        ]
        if not self.is_double:
            reward_positions = reward_positions[:2]
        return reward_positions

    def _reward(self):
        min_steps = (1 + 2 * self.corridor_length)
        redundant_steps = max(0, self.step_count - min_steps)
        max_steps = self.max_steps - min_steps + 1
        cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])
        max_reward = self.reward_values["fake_goal"]
        if hasattr(cell, "is_goal") and cell.is_goal:
            max_reward = self.reward_values["goal"]
        return min(max_reward, max_reward * (1 - min(1, (redundant_steps / max_steps))))

    def _gen_rewards(self, rewards_pos: List[Tuple[int, int]]):
        for i, (x, y) in enumerate(rewards_pos):
            g = Goal()
            self.grid.set(x, y, g)
            g.is_goal = False
            if self.reward_position == i % len(rewards_pos):
                g.is_goal = True

    def render(self, mode='human', close=False, **kwargs):
        reward_positions = self._reward_positions(width=self.width, height=self.height)
        goal = self.grid.get(*reward_positions[self.reward_position])
        assert goal.is_goal
        start_color = goal.color
        goal.color = 'blue'
        ret = super().render(mode, close, **kwargs)
        goal.color = start_color
        return ret

    def close(self):
        if self.grid_render:
            self.grid_render = None


class TMaze(MultiEnv):
    cyclic_order = True
    print_render_buffer = ""
    explored_corners: List[Tuple[int, int]] = None

    @property
    def view_size(self):
        return self.env.agent_view_size

    def __init__(self, corridor_length=3, rounds_pr_side=10, max_steps=None, rnd_order=False, cyclic_order=True,
                 view_size=None):
        self.cyclic_order = cyclic_order
        self.max_steps = max_steps
        self.rnd_order = rnd_order
        self._length_rng = tuple(corridor_length) if hasattr(corridor_length, '__iter__') else (corridor_length,)
        self._rounds_rng = tuple(rounds_pr_side) if hasattr(rounds_pr_side, '__iter__') else (rounds_pr_side,)

        l = random.choice(self._length_rng)
        envs = [SingleTMaze(l, 0, max_steps, view_size=view_size, max_corridor_length=max(self._length_rng)),
                SingleTMaze(l, 1, max_steps, view_size=view_size, max_corridor_length=max(self._length_rng))]
        if self.rnd_order:
            random.shuffle(envs)
        super().__init__(envs, random.choice(self._rounds_rng))

    def reset(self):
        if self.rnd_order:
            random.shuffle(self.schedule)
        elif self.cyclic_order:
            self.schedule = [self.schedule[(i + 1) % len(self.schedule)] for i in range(len(self.schedule))]
        if len(self._length_rng) > 1 or len(self._rounds_rng) > 1:
            if not self.cyclic_order or self.schedule[0][0].reward_position == 0:  # Only change setup at the start of cycle
                l = random.choice(self._length_rng)
                r = random.choice(self._rounds_rng)
                for i, (env, _) in enumerate(self.schedule):
                    env.corridor_length = l
                    self.schedule[i] = (env, r)
        self.print_render_buffer = ""
        return super().reset()

    def step(self, action):
        current_round = int(self.round)
        obs, score, done, info = super().step(action)

        if obs["reward"] != 0:
            if not self.print_render_buffer.endswith("["):
                self.print_render_buffer += f' ,{obs["reward"]}'
            else:
                self.print_render_buffer += f'{obs["reward"]}'
        if obs["round_done"] and self.round == 0:
            self.print_render_buffer += "]\n"
            if not done:
                self.print_render_buffer += f"{self.env.mission}: Rewards=["
        if done:
            self.print_render_buffer += "<END>"

        if current_round == 0:
            score = 0
        else:
            score = score * self.total_rounds / (self.total_rounds - 2)
        return obs, score, done, info

    def on_env_change(self):
        if self.i == 0 and self.round == 0:
            self.print_render_buffer += f"{self.env.mission}: Rewards=["
        self.explored_corners = []

    def seed(self, seed=None):
        random.seed(seed)
        if self.rnd_order:
            if self.round == 0 and self.i == 0:
                random.shuffle(self.schedule)
        super().seed(seed)

    def render(self, mode='human', **kwargs):
        if mode == "print":
            if self.print_render_buffer.endswith("<END>"):
                print(self.print_render_buffer.rstrip("<END>"))
        else:
            return super().render(mode, **kwargs)


def test_one_shot_tmaze():
    import time
    length = 1
    env = SingleTMaze(length, 0)

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
    rounds = 3
    length = 3
    env = TMaze(length, rounds)
    # env: TMaze = gym.make(F"TMaze-{length}x{rounds}-viewsize_3-v0")
    env.seed(1)
    state = env.reset()
    del state["image"]
    print(state)

    # Actions
    # left = 0
    # right = 1
    # forward = 2
    # toggle = 5
    # actions = ([2] * length + [0] + [2] * length) * rounds * 2
    actions = [2] * length + [1] + [2] * length + \
              ([2] * length + [0] + [2] * length) * rounds + \
              ([2] * length + [1] + [2] * length) * (rounds - 1) \
              + ([2] * length + [1] + [2] * length)
    # env.render()
    total_reward = 0
    for a in actions:
        state, reward, done, info = env.step(a)
        # env.render("print")
        env.render()
        time.sleep(.2)
        total_reward += reward
        del state["image"]
        print(reward, done, state)
    print(total_reward)
    assert total_reward >= 1
    time.sleep(1)


if __name__ == '__main__':
    # test_one_shot_tmaze()
    # test_tmaze()
    # env = SingleTMaze(view_size=3, corridor_length=1, max_corridor_length=None, is_double=False)
    # env.render()
    # obs, score, done, info = env.step(random.choice([0, 1, 2]))
    # env.render()
    # time.sleep(10)

    env: TMaze = gym.make("TMazeRnd-2.4-2.10-3-v0")
    env.reset()
    rounds = 0
    while True:
        action = random.choice([0, 1, 2])
        # env.render("print")
        env.render("human")
        # time.sleep(1/20)
        obs, score, done, info = env.step(action)
        if obs["round_done"]:
            rounds += 1
        if done:
            # env.render("print")
            print(f"done. Rounds = {rounds}")
            env.reset()
            rounds = 0
                # break
            print(f"Round: r={env.schedule[0][1]}, g={env.env.grid.width},{env.env.grid.height}")



    # env.view_size =

    # state, reward, done, info = env.step(2)
    # s = env.reset()
    # i = 0
    # print(s["image"][:, :, i])
    # # env.render("human")
    #
    # for action in [2, 0, 2]:
    #     obs, score, done, info = env.step(action)
    #     print(obs["image"][:, :, i])
