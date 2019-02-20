from gym.envs.registration import register

register(
    id='MiniGrid-Choice3x1-color0-v0',
    entry_point='custom_envs.envs:ChoiceEnv3x1_0',
)

register(
    id='MiniGrid-Choice3x1-color1-v0',
    entry_point='custom_envs.envs:ChoiceEnv3x1_1',
)

register(
    id='MiniGrid-Empty-Noise-6x6-v0',
    entry_point='custom_envs.envs:EmptyEnvNoise6x6'
)

register(
    id='MiniGrid-Empty-Noise-8x8-v0',
    entry_point='custom_envs.envs:EmptyEnvNoise'
)

register(
    id='MiniGrid-Empty-Noise-16x16-v0',
    entry_point='custom_envs.envs:EmptyEnvNoise16x16'
)
