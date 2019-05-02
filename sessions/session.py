import datetime
import os
import sys
from typing import Callable

import dill
import shutil
import sys
import time
import traceback
import multiprocessing

from multiprocessing import Process, Queue

from sessions.dirtools import Dir
from git import Repo
from pathlib import Path


class Logger:
    """Copies terminal output to a file"""
    stderr = False

    def __init__(self, filepath: str,  stderr=False):

        self.stderr = stderr
        if self.stderr:
            self.terminal = sys.stderr
        else:
            self.terminal = sys.stdout
        target = Path(filepath)
        self.log = open(target, "a")

    def write(self, message):
        if self.stderr:
            self.terminal.write('\033[91m' + message + '\033[0m')
        else:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def __del__(self):
        try:
            self.stop()
            self.log.close()
        except AttributeError:
            pass

    def stop(self):
        if self.stderr:
            sys.stderr = self.terminal
        else:
            sys.stdout = self.terminal

    def start(self):
        if self.stderr:
            sys.stderr = self
        else:
            sys.stdout = self

    def flush(self):
        self.terminal.flush()


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


def load_session(path, use_backup=True):
    """
    Loads a session
    :param path: The session folder (labelled .ses) containng the session.dill and session.back file
    :param use_backup: If True, will attempt to load backup if main fails
    :return:
    """
    target = Path(path) / "session.dill"

    try:
        with open(target, "rb") as fp:
            return dill.load(fp)[0]
    except Exception as e:
        if use_backup:
            print(f"failed to load {target.name}. Reason:")
            target = target.with_suffix(".back")
            time.sleep(0.1)  # Get nicer print statements
            traceback.print_exc()
            time.sleep(0.1)  # Get nicer print statements
            print("Trying to load backup")
            with open(target, "rb") as fp:
                return dill.load(fp)[0]
        else:
            raise e


class Session:
    """A session represents some work that needs to be done and saved, and possibly paused"""
    is_finished = False  # True when the session has finished all the work
    ignore_uncommited_changes_to_main = True
    runtime = datetime.timedelta()
    _repo_dir = Path(os.path.dirname(os.path.abspath(__file__)))

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
            self._repo_dir = str(repo_dir)
        while ".git" not in os.listdir(self.repo_dir):
            if self.repo_dir == self.repo_dir.parent:
                raise Exception(f"Could not find a .git folder while searching the directory tree")
            self._repo_dir = str(self.repo_dir.parent)
        self.repo = Repo(self.repo_dir)

        self.worker = worker
        self.name = name
        self.ignore_file = ignore_file
        if save_folder is None:
            self._save_folder = f"Experiments/"
        else:
            self._save_folder = str(Path("Experiments") / save_folder)
        os.makedirs(self.save_folder, exist_ok=True)
        self._loggers = [Logger(self.save_folder / "log.txt"), Logger(self.save_folder / "log.txt", stderr=True)]

    @property
    def loggers(self):
        if not hasattr(self, "_loggers"):
            self._loggers = [Logger(self.save_folder / "log.txt"), Logger(self.save_folder / "log.txt", stderr=True)]
        return self._loggers

    @property
    def repo_dir(self):
        return Path(self._repo_dir)

    @property
    def save_folder(self):
        return Path(os.path.dirname(sys.argv[0])) / self._save_folder / (self.name + ".ses")

    def _session_data(self):
        return self.worker, self.repo.head.commit.hexsha, self.is_finished, self.runtime

    def load_results(self):
        """This method is for loading session results after the session has finished"""
        (worker, repo, is_finished) = self.load_data("session")
        commit = repo.index.commit
        if self.repo.index.commit != commit:
            print("Warning: Loaded data belongs to a different commit")

        if not is_finished:
            print("Warning: Loaded data is has not yet finished iterating")
        return worker

    def check_git_status(self):
        """Checks if the there are uncommitted changes to the git head that should be committed before session start"""
        try:
            d = Dir(self.repo.working_dir, exclude_file=self.ignore_file)

            changed_files = [i.a_path for i in self.repo.index.diff(self.repo.head.commit) if
                             not d.is_excluded(Path(self.repo.working_dir) / i.a_path)]
            untracked_files = [f for f in self.repo.untracked_files if
                               not d.is_excluded(Path(self.repo.working_dir) / f)]
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
        except Exception as e:
            print("Encountered exception while checking git status")
            traceback.print_exc()
            return False
        return True

    def save_data(self, filename, data, mode="wb", use_backup=True):
        """Saves data in the data folder"""
        with open(Path(self.save_folder) / f"{filename}.dill", mode) as fp:
            dill.dump(data, fp)
        if use_backup:
            with open(Path(self.save_folder) / f"{filename}.back", mode) as fp:
                dill.dump(data, fp)

    def load_data(self, filename, mode="rb", use_backup=True):
        """loads data from the data folder"""
        assert "b" in mode
        try:
            with open(Path(self.save_folder) / f"{filename}.dill", mode) as fp:
                return dill.load(fp)
        except Exception as e:
            if use_backup:
                print(f"failed to load {filename}.pickle. Reason:")
                time.sleep(0.1)  # Get nicer print statements
                traceback.print_exc()
                time.sleep(0.1)  # Get nicer print statements
                print("Trying to load backup")
                with open(Path(self.save_folder) / f"{filename}.back", mode) as fp:
                    return dill.load(fp)
            else:
                raise e

    def start(self, on_load: Callable[['Session'], None] = None):
        """Starts the session. It will guide the user throug a series of questions about choices for the session
        regarding git status and restarting/overwriting previous sessions"""
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        for logger in self.loggers:
            logger.start()
        print(f"--- {self.name} ---")
        status = self.check_git_status()
        if not status:
            if self.ignore_warnings:
                print("Git status is not clean (Ignored)")
            else:
                print("Git status is not clean. Would you still like to continue with the session? (y/n)")
                response = get_input(valid_inputs=("y", "n"))
                if response == "n":
                    return

        if os.path.exists(self.save_folder / "session.dill"):
            print(f"The save folder already exists. (Path: {self.save_folder})")
            if self.ignore_warnings:
                print(f"Loading session {self.name}")
                response = "l"
            else:
                choices = "Choose one:\n- Restart session and overwrite data folder (r)\n- Load folder (l)\n- Exit (q)"
                print(choices)
                response = get_input(valid_inputs=("r", "l", "q"))
            if response == "l":
                data = self.load_data("session")
                if len(data) == 3:
                    (worker, repo, is_finished) = data
                elif len(data) == 4:
                    (worker, repo, is_finished, runtime) = data
                    self.runtime = runtime
                else:
                    raise AssertionError()

                if isinstance(repo, Repo):

                    try:
                        hexsha = repo.head.commit.hexsha
                    except ValueError:
                        print("(Probably) got error when comparing loaded head with current. This most likely is an "
                              "issue of data stored with an old version of session when loading data on a different PC")
                        traceback.print_exc()
                        hexsha = None
                else:
                    assert isinstance(repo, str)
                    hexsha = repo
                if self.repo.head.commit.hexsha != hexsha:
                    print("WARNING: The loaded session belonged to a different commit and cannot be loaded")
                    print(f"Consider rerunning script after checking out commit({hexsha})")
                    if not self.ignore_warnings:
                        print("Continue? (Y/N)")
                        response = get_input(valid_inputs=("y", "n",))
                        if response == "n":
                            return
                self.worker = worker
                if on_load is not None:
                    print(f"Calling on_load method: {on_load.__name__}")
                    on_load(self)
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
        self.save_data("session", self._session_data())
        shutil.copyfile(Path(sys.argv[0]), Path(self.save_folder) / "script_copy.py")
        print(f"Starting session: {self.name}")
        self._run()

    def _run(self):
        """Don't call explicitly. Instead use start()
        Lets the worker (continue) work until interrupted or finished
        """
        while True:
            try:
                starttime = datetime.datetime.now()
                self.worker.iterate()
                self.runtime += (datetime.datetime.now() - starttime)
                self.save_data("session", self._session_data())
                if self.terminate:
                    print("Session terminated")
                    return
            except StopIteration:
                break
        self.is_finished = True
        self.save_data("session", self._session_data())
        print(f"Session done ({self.name}) in total time: {self.runtime}")
        for logger in self.loggers:
            logger.stop()

    # serialization
    def __getstate__(self):
        state = self.__dict__.copy()
        if "_loggers" in state:
            del state['_loggers']
        return state

    # # serialization
    # def __setstate__(self, state):
    #     for s in ["logger", "_logger", "loggers", "_logger"]:
    #         if s in state:
    #             del state[s]
    #     self.__dict__ = state


class MultiSession(Session):
    """Works like a session except it accepts a list of workers and executes them sequentially"""

    def __init__(self, workers, name, save_folder=None, repo_dir=None, ignore_file='.ignore',
                 ignore_warnings=True, parallel_execution=False):
        self.workers = workers
        self.current_worker = 0
        super().__init__(self, name, save_folder=save_folder, repo_dir=repo_dir, ignore_file=ignore_file,
                         ignore_warnings=ignore_warnings)
        self.parallel_execution = parallel_execution  # TODO: This implementation is quick and dirty hand has potential problems for speical uses
        self.completed = [False] * len(self.worker.workers)
        self.errors = [False] * len(self.worker.workers)


    def _work(self):
        try:
            self.worker.workers[self.current_worker].iterate()
        except StopIteration:
            print(f"Finished worker ({self.current_worker})")
            self.completed[self.current_worker] = True
        except Exception as e:
            print(f"Error in worker ({self.current_worker})")
            traceback.print_exc()
            self.completed[self.current_worker] = True
            self.errors[self.current_worker] = True
        return self.completed[self.current_worker]

    def iterate(self):
        if not all(self.completed):
            if self.parallel_execution:
                for _ in range(len(self.worker.workers)):
                    if not self.completed[self.current_worker]:
                        break
                    self.current_worker = (self.current_worker + 1) % len(self.worker.workers)
                self._work()
                self.current_worker = (self.current_worker + 1) % len(self.worker.workers)
            else:
                done = self._work()
                if done:
                    self.current_worker = (self.current_worker + 1) % len(self.worker.workers)

        else:
            if any(self.errors):
                error_idx = [i for i, err in enumerate(self.errors) if err]
                print(f"WARNING: Unhandled exceptions occurred in thread {error_idx}")
            raise StopIteration()


class MultiThreadedSession(Session):
    """Works like a MultiSession except all workers work concurrently"""

    def __init__(self, workers, name, save_folder=None, repo_dir=None, ignore_file='.ignore', ignore_warnings=True,
                 thread_count: int = None):
        self.workers = workers

        self.status_done = [False] * len(workers)
        self.status_error = [False] * len(workers)
        self.status_working = [False] * len(workers)
        super().__init__(None, name, save_folder=save_folder, repo_dir=repo_dir, ignore_file=ignore_file,
                         ignore_warnings=ignore_warnings)
        self.worker = self
        self.i = 0
        self._thread_count = thread_count
        # raise NotImplementedError("Does not currently work")

    @property
    def thread_count(self):
        return min(len(self.workers),
                   self._thread_count if self._thread_count is not None else multiprocessing.cpu_count())

    def _get_next_worker(self):
        i = int(self.i)
        self.i = (self.i + 1) % len(self.workers)
        for _ in range(len(self.workers)):
            if not self.status_done[i] and not self.status_working[i]:
                return i, self.workers[i]
            i = (i + 1) % len(self.workers)

    def _run(self):
        """Don't call explicitly. Instead use start()
                Lets the worker (continue) work until interrupted or finished
                """
        queue = Queue()

        for i in range(self.thread_count):
            res = self._get_next_worker()
            if res:
                i, worker = res
                self.status_working[i] = True
                p = Process(target=self._process_worker, args=(i, worker, queue))
                p.start()

        while any(self.status_working):
            i, worker, done, error, runtime = queue.get()

            self.workers[i] = worker
            self.status_error[i] = error
            self.status_done[i] = done
            self.status_working[i] = False
            self.runtime += runtime

            res = self._get_next_worker()
            if res:
                i, worker = res
                self.status_working[i] = True
                p = Process(target=self._process_worker, args=(i, worker, queue))
                p.start()
        self.is_finished = True
        self.save_data("session", self._session_data())

        print(f"Session done ({self.name}) in total time: {self.runtime}")
        for logger in self.loggers:
            logger.stop()

    def _process_worker(self, worker_id: int, worker, queue: Queue):
        done, error = False, False
        starttime = datetime.datetime.now()
        try:
            worker.iterate()
        except StopIteration:
            print(f"Finished worker ({worker_id})")
            done = True
        except Exception as e:
            done, error = True, True
            print(f"Error in worker ({worker_id})")
            traceback.print_exc()
        runtime = (datetime.datetime.now() - starttime)
        queue.put((worker_id, worker, done, error, runtime))
