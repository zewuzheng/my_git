import numpy as np
b = np.array([1,2,3])
np.save('test.npy', b)
a = np.load('test.npy')
print(a)
