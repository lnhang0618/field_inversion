"""
Created on Oct 31 2022
@author: lnhang
"""

import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from Generate_Data import Generate_Data
from Solver import Solver

from Objfn import Objfn
from Optimization import Optimization
from Prior import Prior

C_m_type = 'matrix'

N = 32
h = 0.5
eps_0 = 5e-4
T_inf = 50*np.ones((N-1,))

z = np.linspace(0, 1, N+1)
M = 32
T_inf_data = 5*np.ones((M-1,))

beta_prior = np.ones((N-1,))
sigma_prior = 20
C_beta = np.diag(sigma_prior**2*np.ones((N-1,)))

Solver = Solver(N,  h, T_inf)
prior = Prior(beta_prior, C_beta)
data = Generate_Data(M, h, T_inf_data)
data.run(C_m_type=C_m_type,scalar_noise=0.02)

Objfn = Objfn(z, data, Solver, prior)

beta0 = np.ones(N-1)
optimizer = 'Newton-CG'

opt = Optimization(Solver, prior, data, Objfn, beta0, optimizer) 

opt.compute_MAP_properties()
opt.compute_base_properties()
opt.sample_posterior()
opt.compute_true_properties()



f = plt.figure(figsize=(14,4))
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax3 = f.add_subplot(133)

ax1.plot(z[1:-1], opt.T_MAP, 'r', label='MAP')
ax1.plot(data.z_obs[1:-1], opt.T_true, marker='s', markersize=4, 
	fillstyle='none', linestyle='none', color='k', label='True')
ax1.plot(z[1:-1], opt.T_base, '--k', label=' Base')
ax1.set_xlabel(r'$z$')
ax1.set_ylabel(r'$T$')
# ax1.axis([0, 1, 10, 50])
ax1.legend()


ax2.plot(z[1:-1], opt.beta_MAP, 'r', label='MAP')
ax2.plot(data.z_obs[1:-1], opt.beta_true, marker='s', markersize=4, 
	fillstyle='none', linestyle='none', color='k', label='True')
ax2.fill_between(z[1:-1], opt.beta_MAP-2*opt.std, opt.beta_MAP+2*opt.std, 
                 facecolor='black', alpha=0.25)
ax2.axhline(1, linestyle='--', color='k', label='Base')
ax2.set_xlabel('z')
ax2.set_ylabel(r'$\beta$')
'''
if C_m_type=='matrix':
    ax2.axis([0,1,1,1.7])
elif C_m_type=='vector':
    ax2.axis([0,1,-1,3])
else:
    ax2.axis([0,1,-0.5,3])
ax2.legend()
'''


ax3.semilogy(z[1:-1], opt.std, 'r', label='MAP')
ax3.semilogy(z[1:-1], 0.02*np.ones_like(opt.std), marker='s', markersize=4, 
	fillstyle='none', linestyle='none', color='k', label='True')
ax3.semilogy(z[1:-1], sigma_prior*np.ones_like(opt.std), '--k', label='Base')
ax3.legend()
ax3.set_xlabel(r'$z$')
ax3.set_ylabel(r'$\sigma$')
# ax3.axis([0, 1, 1e-2, 1e0])
f.tight_layout()
#f.savefig("./results/fig2(C_m=vector I)(2)_rep")
f.show()


plt.figure()
plt.loglog(opt.conv, 'k')
plt.axhline(opt.J_limit, linestyle='--', color='k')
plt.title('Convergence')
plt.xlabel('Iteration')
plt.ylabel('J')
plt.show()