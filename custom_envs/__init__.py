from gym.envs.registration import register

register(
    id='MiniGrid-Choice3x1-color0-v0',
    entry_point='custom_envs.envs:ChoiceEnv3x1_0',
)

register(
    id='MiniGrid-Choice3x1-color1-v0',
    entry_point='custom_envs.envs:ChoiceEnv3x1_1',
)