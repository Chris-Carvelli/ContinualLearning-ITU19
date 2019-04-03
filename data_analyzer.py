import pandas
import tkinter as tk
from tkinter import filedialog
import os
from sessions.session import load_session
import sessions.session

pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1000)



def get_results_from_session():
    experiments_folder = "Experiments"
    #root = tk.Tk()
    #root.withdraw()
    #file_path = filedialog.askdirectory(initialdir=os.getcwd(), title='Session Folder (.ses)')
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
            if index >= len(session_directories):
                print("Index too large")
            else:
                return load_session(os.getcwd() + "\\" + session_directories[index])
                break

def print_possible_folders(directories_list):
    print("Results folders: ")
    i = 0
    for folder in directories_list:
        print(" "+str(i)+" "+str(folder))
        i = i + 1

def results_to_dataframe(results):
    if isinstance(results, sessions.session.MultiSession):
        workers = results.workers
    elif isinstance(results, sessions.session.Session):
        workers = results.worker
    else:
        workers = [results]

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
