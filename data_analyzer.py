from pathlib import Path

import pandas
import tkinter as tk
from tkinter import filedialog
import os
from sessions.session import Session, load_session
import sessions.session
import seaborn as sns
import matplotlib.pyplot as plt
import click

pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1000)


class SessionResult:
    def __init__(self, _path, _df=None, _name="NoName", _is_single=True):
        self.session: Session = None
        self.split_df = None
        self.session_path = _path
        self.is_single = _is_single
        if _df is None:
            self.load_data()
        else:
            self.df = _df
        self.name = _name

    def load_data(self):
        self.session = load_session(self.session_path)
        df, is_single = results_to_dataframe(self.session)
        self.df = df
        self.is_single = is_single
        if not self.is_single:
            self.split_df = self.df
            self.df = self.df.groupby('generation').mean()


def get_path_to_session(use_explorer):
    experiments_folder = "Experiments"

    if use_explorer:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory(initialdir=Path(os.getcwd()) / "Experiments", title='Session Folder (.ses)')
        return file_path
    else:
        session_directories = []
        for file in os.listdir(experiments_folder):
            path = experiments_folder+"\\"+file
            if os.path.isdir(path):
                session_directories.append(path)

        print_possible_folders(session_directories)

        while True:
            try:
                index = int(input("Select folder by providing index (int) :"))
            except ValueError:
                print("Not an integer! Try again.")
                continue
            else:
                if index >= len(session_directories) and index >= 0:
                    print("Index too large")
                else:
                    return os.getcwd() + "\\" + session_directories[index]
                    break


def print_possible_folders(directories_list):
    print("Results folders: ")
    i = 0
    for folder in directories_list:
        print(" " + str(i) + " " + str(folder))
        i = i + 1


def results_to_dataframe(results):
    if isinstance(results, sessions.session.MultiSession):
        is_single = False
        workers = results.workers
    else:
        workers = [results]
        is_single = True

    d = []
    # ret = (median_score, mean_score, max_score, self.evaluations_used, self.scored_parents)
    experiment_id = 0
    for worker in workers:
        gen = 0
        for line in worker.tuple_results():
            d.append({'run': experiment_id, 'generation': gen, 'median_score': line[0], 'mean_score': line[1],
                      'max_score': line[2], 'evaluations_used': line[3]})
            gen = gen + 1
        experiment_id = experiment_id + 1
    df = pandas.DataFrame(d)
    return df, is_single


def get_ppo_results(path_to_log):
    df2 = pandas.read_csv(path_to_log + "\\log.csv")
    return df2


if __name__ == "__main__":
    import main as mn

    mn.analyze_data()
