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

for h in (1, 2, 4, 6, 8, 10, 12):
    for l in (1, 2, 4, 6, 8, 12, 16, 20, 24):
        register(
            id=f"Copy-{h}x{l}-v0",
            entry_point='custom_envs.envs:Copy',
            kwargs=dict(height=h, length=l)
        )
    register(
        id=f"CopyRnd-{h}-v0",
        entry_point='custom_envs.envs:RandomCopy',
        kwargs=dict(height=h)
    )
for rounds in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
    for length in (1,2,3,4,5,6):
        register(
            id=f"TMaze-{length}x{rounds}-v0",
            entry_point='custom_envs.envs:TMaze',
            kwargs=dict(corrider_length=length, rounds_pr_side=rounds)
        )
