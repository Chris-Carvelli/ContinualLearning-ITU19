# TODO use click in main to make multiple commands
from tests.minigrid.utils import plot

from tests.minigrid.ga import GA
import pickle

if __name__ == "__main__":
    plot('MiniGrid-Empty-Noise-8x8-v0', '10_50_0.005_2')
    # ga = pickle.load(open('ga.pkl', 'rb'))
# import pickle
#
# fp = open(f'process.pickle', 'ab')
# pickle.dump('hello', fp)
# pickle.dump('hello', fp)
# pickle.dump('hello', fp)
