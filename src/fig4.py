'''
Created on Nov 30 2022

@author:lnhang

'''
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
#T_true, T_base, T_ML,T_ML_std=pickle.load(open('../data/figMLdata_case5.p','rb'))
T_true, T_base, T_ML,T_ML_std=pickle.load(open('../data/figMLdata_const_case5.p','rb'))
z=np.linspace(0,1,len(T_true)+2)

f=plt.figure(figsize=(14,4))

ax1=f.add_subplot(121)
ax2=f.add_subplot(122)

# draw T-Z picture
ax1.plot(z[1:-1],T_true,marker='s', markersize=4,
	fillstyle='none', linestyle='none', color='k', label='True')
ax1.plot(z[1:-1],T_ML,'b',label='ML')
ax1.plot(z[1:-1],T_base,'--k',label='Base')
ax1.fill_between(z[1:-1], T_ML-2*T_ML_std, T_ML+2*T_ML_std,
                 facecolor='black', alpha=0.25)
ax1.set_xlabel(r"z")
ax1.set_ylabel(r"T")
ax1.legend()
# draw T-sigma picture
T_pre_error=abs(T_ML-T_true)
ax2.scatter(T_ML_std,T_pre_error,marker='s',color='k')
ax2.set_xlabel(r"$T^{\sigma}$")
ax2.set_ylabel(r"$abs(T_{ML}-T_{true})$")

f.tight_layout()
f.savefig("../results/figML_const_case5")

f.show()









f.show()


