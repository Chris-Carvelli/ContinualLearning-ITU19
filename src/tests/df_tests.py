from collections import OrderedDict

import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

a = np.array(range(10)).reshape((10, 1))
c = np.concatenate((a, np.random.rand(10, 5)), 1)
d = np.concatenate((a, np.random.rand(10, 5)), 1)
e = np.concatenate((c, d), 0)
# print(a)
# print(b)
# print(c)
data = pandas.DataFrame(e,columns = list('xabcde'))
# data.append(pandas.DataFrame(, columns = list('x')))

print(data)

print(data["x"] < 5)
print(data.loc[data["x"] < 5])
# g = data.groupby("x")
# #
# print(g.mean())


print("x" in data)
print("y" in data)

line = sns.lineplot(x="x", y="a", data=data, estimator=None)
# line.axes.ylim(None, 2)
plt.ylim(None, 2)
plt.show()
