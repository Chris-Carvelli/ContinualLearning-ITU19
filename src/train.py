import gym
import dill
import gym_minigrid
import custom_envs
import pandas as pd

from src.ga import GA

N_RUNS = 5

files = [
    'Obstacles',
    'LavaCrossing',
    'KindAll',
    'DistShift'
]

model_builders = {
    'pnn': 'src.modules.PolicyNN:PolicyNN',
    'hnn': 'src.modules.HyperNN:HyperNN'
}

MAX_G = 20
RES_FILE = '../results/comparative_experiment.res'


def main():
    try:
        df = pd.read_pickle(RES_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['gen', 'value', 'type', 'experiment', 'run', 'model'])

    for run in range(N_RUNS):
        for experiment in files:
            for model in model_builders:
                ga = GA(config_file=f'../config_files/{experiment}.cfg', model_builder=model_builders[model])
                for g in range(MAX_G):
                    results = ga.iterate()
                    if results is False:
                        break

                    rows = []

                    for key in results:
                        if 'scored_parents' not in key:
                            rows.append({
                                'value': results[key],
                                'type': key,
                                'experiment': experiment,
                                'run': run,
                                'gen': g,
                                'model': model,
                            })
                    df = df.append(rows, ignore_index=True)
                    df.to_pickle(RES_FILE)
                    dill.dump(ga, open(f'../results/{model}/{experiment}.dill', 'w+b'))


if __name__ == "__main__":
    main()
