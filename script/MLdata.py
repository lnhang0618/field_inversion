'''
Created on Nov 17 2022

@author:lnhang
'''

import sys
sys.path.append('../src')

import numpy as np

from Generate_Data import Generate_Data
from Objfn_const_beta import Objfn
from Solver import Solver
from Prior import Prior
from Optimization_const_beta import Optimization

import pickle
from scipy.optimize import minimize

np.random.seed(100)
C_m_type = "matrix"
N=32
h=0.5
z=np.linspace(0,1,N+1)
M=32
beta0=0.5
optimizer='Newton-CG'

beta_prior = np.ones((N-1,))

sigma_prior=1

sigma_prior_lst = [20, 2, 1, 1, 0.5, 1, 1, 1 ,1, 0.8]

#T_inf_l=[28,55,35+20*np.sin(2*np.pi*z[1:-1]),35-15*z[1:-1],15+5*np.cos(np.pi*z[1:-1])]

T_inf_lst=[]
T_obs_lst=[]
beta_MAP_lst=[]
std_lst=[]

#T_inf_lst = []
#T_true_lst = []
#beta_true_lst = []

for i, T_inf in enumerate(range(5, 51, 5)):
    sigma_prior = sigma_prior_lst[i]
    C_beta = np.diag(sigma_prior ** 2 * np.ones((N - 1,)))

    solver = Solver(N,  h, T_inf)
    prior = Prior(beta_prior, C_beta)
    data = Generate_Data(M, h, T_inf)
    data.run(C_m_type=C_m_type, scalar_noise=0.02)

    objfn = Objfn(z, data, solver, prior)

    beta0 = np.ones((N - 1,))
    optimizer = 'Newton-CG'

    opt = Optimization(solver, prior, data, objfn, beta0, optimizer)

    opt.compute_MAP_properties()
    opt.compute_true_properties()
    opt.compute_base_properties()
    opt.sample_posterior()

    T_inf_lst.append(T_inf * np.ones((N - 1)))
    T_obs_lst.append(opt.data.T_obs)
    beta_MAP_lst.append(opt.beta_MAP)
    std_lst.append(opt.std)

pickle.dump((T_inf_lst, T_obs_lst, beta_MAP_lst, std_lst), open('../data/ML_const_1time.p', 'wb'))


'''
    objfn=Objfn(z,data,solver,prior)

    beta0=np.ones(N-1,)

    opt = Optimization(solver, prior, data, objfn, beta0, optimizer)

    opt.compute_MAP_properties()
    opt.compute_true_properties()
    opt.compute_base_properties()
    opt.sample_posterior()
'''
   # T_inf_lst.append(T_inf * np.ones((N - 1)))
   # T_obs_lst.append(opt.data.T_obs)
   # beta_MAP_lst.append(opt.beta_MAP)
   # std_lst.append(opt.std)


#pickle.dump((T_inf_lst, T_true_lst, beta_true_lst), open('../data/MLtest.p','wb'))