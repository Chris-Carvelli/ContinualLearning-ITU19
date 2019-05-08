from src.modules.CopyNTM import CopyNTM
import numpy as np

if __name__ == '__main__':

    for _ in range(10):
        ntm = CopyNTM(2, memory_unit_size=4)
        for _ in range(10):
            # x = np.random.randint(0, 2, (4,))
            x = np.random.randn(4)*0.05 + 0.5
            ntm.reset()
            ntm.evolve(0.5)
            print(x, ntm(x))
        print("-----------")