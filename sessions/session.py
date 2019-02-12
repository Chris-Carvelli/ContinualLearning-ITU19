import os
import pickle
import shutil
import sys
import threading
import time
import traceback

from sessions.dirtools import Dir
from git import Repo
from pathlib import Path


def get_input(valid_inputs=("y", "n")):
    """
    Queries the user for a terminal input which must be one of the valid inputs specified
    :param valid_inputs: A iterable of string the represents valid inputs
    :return:
    """
    res = input().lower()
    while res not in valid_inputs:
        print(f'Valid responses: ({"/".join(valid_inputs)})')
        res = input().lower()
    return res


class Session:
    """A session represents some work that needs to be done and saved, and possibly paused"""
    repo_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    is_finished = False
    terminate = False

    def __init__(self, worker, name, save_folder=None, repo_dir=None, ignore_file='.ignore'):
        """
        :param worker: A object that implements the __next__() and throws a StopIteration when finished.
        :param name: The name of the session. Unless otherwise specified it will also be the name of the data folder
        :param save_folder: The folder where the data is stored. Default is a folder in the same directory as the
                            main script with same name as the session
        :param repo_dir: The repository in for which the commit state at session runtime is stored if the session is
                            to be restarted at a later time
        :param ignore_file: A file similar to a .gitignore that allows you to specify certain files which can be
                            modified without affecting the result of the worker
        """
        if repo_dir is not None:
            self.repo_dir = repo_dir
        self.worker = worker
        self.name = name
        self.ignore_file = ignore_file
        self.save_folder = save_folder
        if self.save_folder is None:
            self.save_folder = Path(os.path.dirname(sys.argv[0])) / self.name
        self.repo = Repo(self.repo_dir)
        self.session_data = lambda: (self.worker, self.repo, self.is_finished)

    def check_git_status(self):
        """Checks if the there are uncommitted changes to the git head that should be committed before session start"""
        d = Dir(self.repo.working_dir, exclude_file=self.ignore_file)
        changed_files = [i.a_path for i in self.repo.index.diff(self.repo.head.commit) if
                         not d.is_excluded(Path(self.repo.working_dir) / i.a_path)]
        untracked_files = [f for f in self.repo.untracked_files if not d.is_excluded(Path(self.repo.working_dir) / f)]
        if len(changed_files) + len(untracked_files) > 0:
            print("The following files were untracked or had uncommitted changes:")
            for f in changed_files + untracked_files:
                print("- " + f)
            return False
        return True

    def save_data(self, filename, data, mode="wb", use_backup=True):
        """Saves data in the data folder"""
        with open(Path(self.save_folder) / f"{filename}.pickle", mode) as fp:
            pickle.dump(data, fp)
        if use_backup:
            with open(Path(self.save_folder) / f"{filename}.back", mode) as fp:
                pickle.dump(data, fp)

    def load_data(self, filename, mode="rb", use_backup=True):
        """loads data from the data folder"""
        assert "b" in mode
        try:
            with open(Path(self.save_folder) / f"{filename}.pickle", mode) as fp:
                return pickle.load(fp)
        except Exception as e:
            if use_backup:
                print(f"failed to load {filename}.pickle. Reason:")
                time.sleep(0.1)  # Get nicer print statements
                traceback.print_exc()
                time.sleep(0.1)  # Get nicer print statements
                print("Trying to load backup")
                with open(Path(self.save_folder) / f"{filename}.back", mode) as fp:
                    return pickle.load(fp)
            else:
                raise e

    def _termination_input(self):
        """A subroutine to check for user input"""
        m = "input (q) to terminate session after next iteration"
        print(m)
        res = input().lower()
        while res != "q" and not self.is_finished:
            print(m)
            res = input().lower()
            time.sleep(0.5)
        if res == "q":
            self.terminate = True

    def start(self):
        """Starts the session. It will guide the user throug a series of questions about choices for the session
        regarding git status and restarting/overwriting previous sessions"""
        status = self.check_git_status()
        if not status:
            print("Git status is not clean. Would you still like to continue with the session? (y/n)")
            response = get_input(valid_inputs=("y", "n"))
            if response == "n":
                return
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        else:
            print(f"The save folder already exists. (Path: {self.save_folder})")
            choices = "Choose one:\n- Restart session and overwrite data folder (r)\n- Load folder (l)\n- Exit (q)"
            print(choices)
            response = get_input(valid_inputs=("r", "l", "q"))
            if response == "l":
                (iterator, repo, is_finished) = self.load_data("session")
                if is_finished:
                    print("Loaded session is already finished.")
                    return
                commit = repo.head.commit
                if self.repo.head.commit != commit:
                    print("The loaded session belonged to a different commit and cannot be loaded")
                    print(f"Before rerunning script please checkout commit({commit}): {commit.message}")
                    return
                else:
                    self.worker = iterator
                    self._run()
                    return
            elif response == "r":
                print("Confirm (y/n)")
                if get_input(("y", "n")) == "y":
                    shutil.rmtree(self.save_folder)
                    os.makedirs(self.save_folder)
                else:
                    return
            else:
                return
        self.save_data("session", self.session_data())
        self._run()

    def _run(self):
        """Don't call explicitly. Instead use start()
        Lets the worker (continue) work until interrupted or finished
        """
        t = threading.Thread(target=self._termination_input)
        # TODO: Fix so that this thread terminates after session is complete
        t.start()
        while True:
            try:
                i = self.worker.__next__()
                if i is not None:
                    print(i)
                self.save_data("session", self.session_data())
                if self.terminate:
                    print("Session terminated")
                    return
            except StopIteration:
                break
        self.is_finished = True
        print(self.session_data()[2])
        self.save_data("session", self.session_data())
        print("Session done")

