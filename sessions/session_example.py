"""
This is a test document to test out the possibility of using git commits when saving data from training runs so
the experiments can be re-run with the same settings/code as they the data was generated
"""
import os
import random
from pathlib import Path

from sessions.session import *


class MyExperiment:
    def __init__(self, name="#"):
        self.name = name
        self.current = []
        self.count = 0

    def iterate(self):
        # if self.count >= 5:
        if random.random() > .9:
            raise StopIteration()
        self.current += [x for x in range(100)]
        # if self.count > 2:
        #     s = 0
        #     while True:
        #         s += 1
        self.count += 1
        # if self.count > 1:
        #     raise AssertionError()
        print(f"{self.name}: {self.count}")
        time.sleep(0.2)
        return sum(self.current)


if __name__ == '__main__':
    # S = Session(MyExperiment(), "TestSession")
    # S.start()

    # # Example of MultiSession
    ms = MultiSession([MyExperiment("0"), MyExperiment("1"), MyExperiment("2")], "TestSessions", parallel_execution=True)
    ms.start()

    # Example of MultiSession
    # mts = MultiThreadedSession([MyExperiment(), MyExperiment(), MyExperiment(), MyExperiment(), MyExperiment()], "MTS")
    # mts.start()
