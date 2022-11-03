import numpy as np

a1 = np.asarray([1, 2, 3])
a2 = np.asarray([0, 4, 5])

mse = (np.square(a1 - a2)).mean(axis=0)
print(mse)