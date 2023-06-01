import numpy as np

def H(T):
        return np.interp(z_obs[1:-1],z[1:-1],T)
def compute_H_matrix(num,N):
        H_matrix = np.zeros((num-1,N-1))
        T_test=np.zeros(N-1)
        for i in range(N-1):
            T_test[i]=1
            H_matrix[:,i]=H(T_test)
            print(H_matrix)
            T_test[:]=0
        return H_matrix

z=[0,1,2.2,3,4,5]
z_obs=[0.5,1.5,2.5,3.5,4.4,5.5]
H_mtx=compute_H_matrix(5,5)
print(H_mtx)