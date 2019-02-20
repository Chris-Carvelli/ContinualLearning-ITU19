"""
This is a test document to test out the possibility of using git commits when saving data from training runs so
the experiments can be re-run with the same settings/code as they the data was generated
"""
import os
from pathlib import Path

from sessions.session import Session


def my_experiment():
    yield "Starting experiment"
    for x in range(10):
        yield x


class MyExperiment:
    def __init__(self):
        self.current = []
        self.count = 0

    def __next__(self):
        if self.count >= 10:
            raise StopIteration()
        self.current += [x for x in range(1000)]
        self.count += 1
        return sum(self.current)


S = Session(MyExperiment(), "TestSession")
S.start()