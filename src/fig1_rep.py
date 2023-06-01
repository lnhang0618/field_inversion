'''
Created on Oct 16 2022

@author:lnhang

'''
import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from Generate_Data import Generate_Data
from Solver import Solver

'''
set n,h,T_inf
'''
n=30
h=0.5

z=np.linspace(0,1,n+1)

plt.figure()
'''
T_inf=20
if T_inf==20:
'''
#for T_inf in np.linspace(5,50,5):
for T_inf in range(50,51,5):
    Data1=Generate_Data(n,h,T_inf)
    Data2=Solver(n,h,T_inf)
    
    T_true=Data1.get_T_true()
    T_Base=Data2.get_T_base()
    
    plt.plot(z[1:n],T_true,marker='s',linestyle="none",color='#E4292E',label="True" if T_inf==5 else None)
    plt.plot(z[1:n],T_Base,color='#3979F2',label="Base" if T_inf==5 else None)
plt.xlim(0.0,1.0)
plt.ylim(15,51)
plt.xlabel("position z")
plt.ylabel("output T")
plt.legend()
# plt.tight_layout()
plt.savefig("../results/fig1(T=50)")
plt.show()
