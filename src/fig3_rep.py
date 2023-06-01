'''
Created on Nov 5 2022

@author:lnhang

'''


import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from Generate_Data import Generate_Data
from Objfn import Objfn
from Optimization import Optimization
from Prior import Prior
from Solver import Solver

np.random.seed(100)
C_m_type = "matrix"
N= 32
h=0.5
z=np.linspace(0,1,N+1)
M=32
beta0=0.5
optimizer='Newton-CG'



#Solver = Solver(N,  h, T_inf)
#prior = Prior(beta_prior, C_beta)
#data = Generate_Data(M, h, T_inf_data)
#data.run(C_m_type=C_m_type,scalar_noise=0.03)

#Objfn = Objfn(z, data, Solver, prior)

beta_prior=np.ones((N-1,))
sigma_prior=0.8
C_beta=np.diag(sigma_prior**2*np.ones((N-1,)))

prior = Prior(beta_prior, C_beta)

f=plt.figure(figsize=(14,4))

ax1=f.add_subplot(131)
ax2=f.add_subplot(132)
ax3=f.add_subplot(133)


for i,T_inf in enumerate(range(5,51,5)):
    print(T_inf)
    data=Generate_Data(M,h,T_inf)
    data.run(C_m_type,scalar_noise=0.02)
    solver=Solver(N,h,T_inf)
    T_true = data.get_T_true()
    beta_r_true=solver.get_beta_r_true(T_true)
    beta_c_true=solver.get_beta_c_true(T_true)

    
    ax1.plot(T_true,beta_r_true,"k",label="True" if i==0 else "")
    ax2.semilogy(T_true,beta_c_true,"k",label="True" if i == 0 else '')
    objfn = Objfn(z, data, solver, prior)
    opt = Optimization(solver, prior, data, objfn, beta0, optimizer) 
    opt.compute_MAP_properties()
    opt.sample_posterior(int(1e6))
    
    std=opt.std
    beta_MAP=opt.beta_MAP
    beta_r_MAP=opt.beta_r_MAP
    beta_c_MAP=opt.beta_c_MAP
    T_MAP=solver.get_T_beta(beta_MAP)
    
    ax1.errorbar(T_MAP,beta_r_MAP,yerr=std,marker='o',fillstyle='none',linestyle='none',
                 mec="r",ecolor='0.5',ms=4,capsize=3,label='MAP' if i==0 else '')
    ax2.plot(T_MAP,beta_c_MAP,marker='o',fillstyle='none',linestyle='none',mec='r',
             ms=4,label='MAP' if i==0 else'')
    ax3.plot(T_MAP,std,marker="o",fillstyle='none',linestyle='none',color='r',ms=4
             ,label='MAP' if i==0 else '')
    ax3.plot(T_true,[0.02]*(len(T_true)),linestyle='-',color='k',label="True" if i ==0 else '')
    
    
ax1.set_xlabel(r'$T$')
ax1.set_ylabel(r"$\beta_r$")

ax2.set_xlabel(r"$T$")
ax2.set_ylabel(r"$\beta_c$")

ax3.set_xlabel(r"$T$")
ax3.set_ylabel(r"$\sigma$")

if C_m_type=='matrix':
    ax1.axis([0, 50, 0.2, 1.8])
elif C_m_type=='vector':
    ax1.axis([0, 50, -1, 2.5])
elif C_m_type=='scalar':
    ax1.axis([0, 50, -0.5, 2.5])

ax2.axis([0,50,1e-3,1e1])

ax1.legend()
ax2.legend()
ax3.legend()
f.savefig('./results/fig3(C_m=full)_rep')
f.show()
