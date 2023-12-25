import time
import numpy as np

a = np.zeros(100000)
print(a.shape)

s = time.time()

for i in range(10000,30000):
    a[i] = a[-1]
    a = np.delete(a, -1)

e = time.time()
print(a.shape)
print("Time take: ", e-s)

print("="*40)

a = np.zeros(100000)
print(a.shape)

s = time.time()

for i in range(10000,30000):
    a[i] = a[-1]
    a = np.delete(a, -1)

e = time.time()
print(a.shape)
print("Time take: ", e-s)