import pickle
import pandas
import tkinter as tk
from tkinter import filedialog
import os
import sessions.session

pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1000)


def get_results_from_session():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askdirectory(initialdir=os.getcwd(), title='Session Folder (.ses)')

    path = file_path + '/session.pickle'
    print("Loading from: ", path)
    data = pickle.load(open(path, "rb"))
    results = data[0].load_results()
    return results


def results_to_dataframe(results):

    if isinstance(results, sessions.session.MultiSession):
        workers = results.workers
    elif isinstance(results, sessions.session.Session):
        workers = results.worker
    d = []
    # ret = (median_score, mean_score, max_score, self.evaluations_used, self.scored_parents)
    experiment_id = 0
    for worker in workers:
        gen = 0
        for line in worker.results:
            d.append({'run': experiment_id, 'generation': gen, 'median_score': line[0], 'mean_score': line[1],
                      'max_score': line[2], 'evaluations_used': line[3]})
            gen = gen + 1
        experiment_id = experiment_id + 1
    df = pandas.DataFrame(d)
    return df
