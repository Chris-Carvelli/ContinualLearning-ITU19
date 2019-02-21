from tests.minigrid.ga import GA
import gym_minigrid

for runs in experiments:
        print('main')

        ga = GA()

        pickle.dump(ga, open('ga.pickle', 'wb'))
        fp = open('process.pickle', 'ab')
        while True:
                try:
                        res = ga.optimize()
                        print(f'done with generation{res[0]}')
                        pickle.dump(res, fp)
                except StopIteration:
                        break
        fp.close()
