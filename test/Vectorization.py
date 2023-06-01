import numpy as np

A=np.ones((5,5))
print(np.sum(A**2,1).reshape(-1,1))
print(np.sum(A**2,1))
B=np.sum(A**2,1).reshape(-1,1)+np.sum(A**2,1)
print(B)