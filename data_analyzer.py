import pickle
import pandas

path = 'TestSessions.ses/session.pickle'
data = pickle.load(open(path, "rb"))
experiment = data[0].load_results()

for worker in experiment.workers:
    print(worker.results)