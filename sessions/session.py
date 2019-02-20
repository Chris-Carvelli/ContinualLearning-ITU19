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
    repo_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    is_finished = False  # True when the session has finished all the work
    ignore_uncommited_changes_to_main = True

    def __init__(self, worker, name, save_folder=None, repo_dir=None, ignore_file='.ignore', ignore_warnings=True):
        """
        :param worker: A object that implements the work() methods which throws a StopIteration when finished.
        :param name: The name of the session. Unless otherwise specified it will also be the name of the data folder
        :param save_folder: The folder where the data is stored. Default is a folder in the same directory as the
                            main script with same name as the session
        :param repo_dir: The repository in for which the commit state at session runtime is stored if the session is
                            to be restarted at a later time
        :param ignore_file: A file similar to a .gitignore that allows you to specify certain files which can be
                            modified without affecting the result of the worker
        """
        self.ignore_warnings = ignore_warnings
        self.terminate = False
        if repo_dir is not None:
            self.repo_dir = repo_dir
        while ".git" not in os.listdir(self.repo_dir):
            self.repo_dir = self.repo_dir.parent

        self.repo = Repo(self.repo_dir)

        self.worker = worker
        self.name = name
        self.ignore_file = ignore_file
        self.save_folder = save_folder
        if self.save_folder is None:
            self.save_folder = os.path.dirname(sys.argv[0])
        self.save_folder = Path(self.save_folder) / self.name
        self.save_folder = Path(self.save_folder).with_suffix(".ses")
        self.session_data = lambda: (self.worker, self.repo, self.is_finished)

    def load_results(self):
        """This method is for loading session results after the session has finished"""
        (worker, repo, is_finished) = self.load_data("session")
        commit = repo.head.commit
        if self.repo.head.commit != commit:
            print("Warning: Loaded data belongs to a different commit")

        if not is_finished:
            print("Warning: Loaded data is has not yet finished iterating")
        return worker

    def check_git_status(self):
        """Checks if the there are uncommitted changes to the git head that should be committed before session start"""
        d = Dir(self.repo.working_dir, exclude_file=self.ignore_file)

        changed_files = [i.a_path for i in self.repo.index.diff(None) if
                         not d.is_excluded(Path(self.repo.working_dir) / i.a_path)]
        untracked_files = [f for f in self.repo.untracked_files if not d.is_excluded(Path(self.repo.working_dir) / f)]
        dirty_files = changed_files + untracked_files
        if self.ignore_uncommited_changes_to_main:
            target = sys.argv[0].replace(Path(self.repo_dir).as_posix() + "/", "")
            if target in dirty_files:
                dirty_files.remove(target)
        if len(dirty_files) > 0:
            print("The following files were untracked or had uncommitted changes:")
            for f in dirty_files:
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

    def start(self):
        """Starts the session. It will guide the user throug a series of questions about choices for the session
        regarding git status and restarting/overwriting previous sessions"""
        status = self.check_git_status()
        if not status:
            if self.ignore_warnings:
                print("Git status is not clean (Ignored)")
            else:
                print("Git status is not clean. Would you still like to continue with the session? (y/n)")
                response = get_input(valid_inputs=("y", "n"))
                if response == "n":
                    return
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        else:
            print(f"The save folder already exists. (Path: {self.save_folder})")
            if self.ignore_warnings:
                print("Loading session")
                response = "l"
            else:
                choices = "Choose one:\n- Restart session and overwrite data folder (r)\n- Load folder (l)\n- Exit (q)"
                print(choices)
                response = get_input(valid_inputs=("r", "l", "q"))
            if response == "l":
                (worker, repo, is_finished) = self.load_data("session")
                # if is_finished:
                #     self.worker = worker
                #     print("Loaded session is already finished.")
                #     return
                commit = repo.head.commit
                if self.repo.head.commit != commit:
                    print("The loaded session belonged to a different commit and cannot be loaded")
                    print(f"Before rerunning script please checkout commit({commit}): {commit.message}")
                    return
                else:
                    self.worker = worker
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
        shutil.copyfile(Path(sys.argv[0]), Path(self.save_folder) / "script_copy.py")
        self._run()

    def _run(self):
        """Don't call explicitly. Instead use start()
        Lets the worker (continue) work until interrupted or finished
        """
        while True:
            try:
                i = self.worker.iterate()
                self.save_data("session", self.session_data())
                if self.terminate:
                    print("Session terminated")
                    return
            except StopIteration:
                break
        self.is_finished = True
        self.save_data("session", self.session_data())
        print("Session done")
