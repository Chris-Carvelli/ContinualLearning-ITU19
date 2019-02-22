from tests.minigrid.ga import GA
import gym_minigrid
from sessions.session import Session
import os

class Experiment:
    def __init__(self, config_file_path, number_of_runs):
        self.config_file_path = config_file_path
        self.number_of_runs = number_of_runs
        self.ga = GA(config_file_path)

    def iterate(self):
        runs = 0
        while runs < self.number_of_runs:
            #TODO talk with rasmus about this, when will session end, does it freeze the thread, how is it being logged
            S = Session(self.ga, "Experiment: "+str(runs))
            S.start()
            runs = runs + 1
            #TODO Reset GA

first_experiment = Session(Experiment('config_files/config_one', 5), "Experiment: ")
first_experiment.start()