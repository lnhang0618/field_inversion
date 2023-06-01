import sys

sys.path.append('../src')

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
T_inf = 5*np.ones((N-1,))

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

color=["#F7903D","#4D85BD","#59A95A"]
plt.figure()
plt.plot(z[1:-1],data.T_obs,marker='s',linestyle="none",color=color[0],label="True")
plt.plot(z[1:-1],opt.T_base,color=color[1],label="Base")
plt.plot(z[1:-1],opt.T_MAP,color=color[2],label="MAP")

plt.xlabel("position x")
plt.ylabel("output T")
plt.grid(linestyle="-.")

plt.xlim(0.0,1.0)
plt.ylim(0.00,0.33)

plt.tight_layout()
plt.legend()
plt.savefig("../results/Fig2(T=5)")
