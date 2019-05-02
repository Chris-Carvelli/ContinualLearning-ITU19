from src.ga import *

if __name__ == '__main__':
    # plot sigma_strategies
    class A:
        def __init__(self, g):
            self.g = g
            self.sigma = 1

    import matplotlib.pyplot as plt
    n = 2000
    objs = [A(g) for g in range(n)]
    x = np.array([i for i in range(n)])
    plot_names = [
        "cyclic1000-0.01",
        'linear1000-0.1',
        'linear1000-0.01',
        # 'linear10000-0.001',
    ]

    for name, f in sigma_strategies.items():
        if name not in plot_names:
            continue
        y = np.array([f(obj) for obj in objs])
        plt.plot(x, y)

        plt.ylabel('name')
    plt.show()