from gym.envs.registration import register

# Avoid importing local env by registering env in a function call
def _register_envs():

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
        for l in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 100):
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
        register(
            id=f"CopyRnd-{h}-10-v0",
            entry_point='custom_envs.envs:RandomCopy',
            kwargs=dict(height=h, min_length=1, max_length=12)
        )


    for l1, l2, r1, r2 in [(2, 4, 2, 10), (2, 6, 2, 10)]:
        register(
            id=f"TMazeRnd-{l1}.{l2}-{r1}.{r2}-3-v0",
            entry_point='custom_envs.envs:TMaze',
            kwargs=dict(corridor_length=range(l1, l2 + 1), rounds_pr_side=range(r1, r2 + 1), view_size=3, cyclic_order=False)
        )
    register(
        id=f"TMaze-2.3x4.6-viewsize_3-v0",
        entry_point='custom_envs.envs:TMaze',
        kwargs=dict(corridor_length=range(2, 3+1), rounds_pr_side=range(4, 6 + 1), view_size=3)
    )
    register(
        id=f"TMaze-2x4-3-UnevenRounds-v0",
        entry_point='custom_envs.envs:TMaze',
        kwargs=dict(corridor_length=2, rounds_pr_side=4, view_size=3, uneven_rounds=True)
    )
    register(
        id=f"TMaze-2x4-3-UnevenRounds-x2-v0",
        entry_point='custom_envs.envs:TMaze',
        kwargs=dict(corridor_length=2, rounds_pr_side=4, view_size=3, uneven_rounds=True, repeat=2)
    )

    for rounds in (1, 2, 3, 4, 5, 6, 10, 20, 50, 100, 200, 500, 1000):
        for length in (1,2,3,4,5,6):
            register(
                id=f"TMaze-{length}x{rounds}-v0",
                entry_point='custom_envs.envs:TMaze',
                kwargs=dict(corridor_length=length, rounds_pr_side=rounds)
            )
            for v in [3, 5]:
                register(
                    id=f"TMaze-{length}x{rounds}-viewsize_{v}-v0",
                    entry_point='custom_envs.envs:TMaze',
                    kwargs=dict(corridor_length=length, rounds_pr_side=rounds, view_size=v)
                )
                register(
                    id=f"DoubleTMaze-{length}x{rounds}-{v}-v0",
                    entry_point='custom_envs.envs:TMaze',
                    kwargs=dict(corridor_length=length, rounds_pr_side=rounds, view_size=v, double=True)
                )
            for max_steps in [6, 12, 20, 50]:
                register(
                    id=f"TMaze-{length}x{rounds}x{max_steps}-v0",
                    entry_point='custom_envs.envs:TMaze',
                    kwargs=dict(corridor_length=length, rounds_pr_side=rounds, max_steps=max_steps)
                )

    register(
        id=f"SingleTMaze-v0",
        entry_point='custom_envs.envs:SingleTMaze'
    )

_register_envs()