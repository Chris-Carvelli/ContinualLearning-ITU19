"""
This is a test document to test out the possibility of using git commits when saving data from training runs so
the experiments can be re-run with the same settings/code as they the data was generated
"""
import os
from pathlib import Path

from sessions.session import *

class MyExperiment:
    def __init__(self):
        self.current = []
        self.count = 0

    def iterate(self):
        if self.count >= 3:
            raise StopIteration()
        self.current += [x for x in range(100)]
        # if self.count > 2:
        #     s = 0
        #     while True:
        #         s += 1
        self.count += 1
        print(self.count)
        time.sleep(1)
        return sum(self.current)

if __name__ == '__main__':


    # S = Session(MyExperiment(), "TestSession")
    # S.start()

    # # Example of MultiSession
    # ms = MultiSession([MyExperiment(), MyExperiment(), MyExperiment()], "TestSessions")
    # ms.start()

    # Example of MultiSession
    mts = MultiThreadedSession([MyExperiment(), MyExperiment(), MyExperiment(), MyExperiment(), MyExperiment()], "MTS")
    mts.start()


