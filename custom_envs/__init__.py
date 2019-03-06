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

for copy_size in (1, 2, 4, 6, 8, 10, 12):
    for length in (1, 2, 4, 6, 8, 12, 16, 20, 24):
        register(
            id=f"Copy-{copy_size}x{length}-v0",
            entry_point='custom_envs.envs:Copy',
            kwargs=dict(height=copy_size, length=length)
        )
    register(
        id=f"CopyRnd-{copy_size}-v0",
        entry_point='custom_envs.envs:RandomCopy',
        kwargs=dict(height=copy_size)
    )
