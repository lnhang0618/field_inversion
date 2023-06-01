import numpy as np

T=[[1,2,3],[2,3,4]]
G=np.asarray(T)
std=np.std(G,axis=0)
print(std)
