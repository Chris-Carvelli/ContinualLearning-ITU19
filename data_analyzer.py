import pandas
import tkinter as tk
from tkinter import filedialog
import os
from sessions.session import load_session
import sessions.session
import seaborn as sns
import matplotlib.pyplot as plt
import click
from src.ga import GA

pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1000)



def get_results_from_session(use_explorer):
    experiments_folder = "Experiments"

    if use_explorer:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory(initialdir=os.getcwd(), title='Session Folder (.ses)')
        return load_session(file_path)
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
        print(" "+str(i)+" "+str(folder))
        i = i + 1

def results_to_dataframe(results):
    print(results)
    if isinstance(results, sessions.session.MultiSession):
        workers = results.workers
        is_single = False
    else:
        print("ERROR! Something is weird")
        workers = [results]
        is_single = True

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
    return df, is_single

def get_ppo_results(path_to_log):
    df2 = pandas.read_csv(path_to_log+"\\log.csv")
    return df2

@click.command()
@click.option('--ppo_results', default='', help='Path to .csv file containing results of PPO')
@click.option('--session_results', default='', help='Path to .ses folder to analyze')
def analyze_data(ppo_results, session_results):
    # Get path to session
    # If not provided as argument propmt the user
    if not session_results:
        # True -> Use explorer
        # False -> Use terminal
        res_path = get_results_from_session(False)
    else:
        res_path = session_results

    # Get session and put the data in data frame
    result_session = load_session(res_path)
    df, is_single = results_to_dataframe(result_session)

    # Plot runs against each other if it's a multi-session
    if not is_single:
        sns.lineplot(x="generation", y="max_score", hue="run", data=df).set_title("Max Score per run")
        plt.show()

    # ppo_results = "C:\\Users\\Luka\\Documents\\Python\\minigrid_rl\\torch-rl\\storage\\DoorKey"
    # Get results as dataframe from minigrid_rl. You need to provide the path of the log
    if ppo_results:
        df2 = get_ppo_results(ppo_results)
        # print(df2.columns.values)
        print(df.columns.values)
        print(df2.columns.values)
        # print(df2)
        print(df2)
        # df = df.set_index('generation').join(df2.set_index('update'))

    # Combine all runs into one and do an average of the results if is multisession
    if not is_single:
        df = df.groupby('generation').mean()
    # Plot combined data
    joined_data = [df['mean_score'], df['max_score'], df['median_score']]
    joined_data_plot = sns.lineplot(data=joined_data).set_title("Max, Mean and Median Score Averaged over all runs")
    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.legend(['mean', 'max', 'median'])

    plt.show()

if __name__ == "__main__":
    analyze_data()

