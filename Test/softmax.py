import numpy as np

a = np.array([1, 2, 3])

b = np.exp(a) / np.sum(np.exp(a))

print(b)
print(sum(b))