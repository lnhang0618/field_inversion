'''
Created on Nov 1 2022

@author:lnhang
'''
import numpy as np
# from numdifftools.core import Hessian
# from tensorflow import einsum

class Objfn:
    def __init__(self,z,data,Solver,prior):
        self.N=len(z)-1
        self.z=z
        self.dz=z[1]-z[0]
        self.data=data
        self.Solver=Solver
        self.prior=prior
        
        self.H_matirx=self.compute_H_matrix()
    
    def H(self,T):
        return np.interp(self.data.z[1:-1],self.data.z_obs[1:-1],T,left=0,right=0)
    
    def compute_H_matrix(self):
        H_matrix = np.zeros((self.data.num-1,self.N-1))
        T_test=np.zeros(self.N-1)
        for i in range(self.N-1):
            T_test[i]=1
            H_matrix[:,i]=self.H(T_test)
            T_test[:]=0
        return H_matrix
    
    def compute_dRdT(self,beta):
        T=self.Solver.get_T_beta(beta)
        main=1+2*self.dz**2*5e-4*beta*T**3
        lower=-0.5*np.ones((self.N-2,))
        upper=-0.5*np.ones((self.N-2,))
        dRdt=np.diag(main)+np.diag(lower,k=-1)+np.diag(upper,k=1)
        return dRdt
    
    def compute_dRdbeta(self,beta):
        T=self.Solver.get_T_beta(beta)
        main = 0.5*self.dz**2*5e-4*(T**4-self.Solver.T_inf**4)
        dRdbeta=np.diag(main)
        return dRdbeta   
    
    def compute_dJdT(self,beta): 
        T=self.Solver.get_T_beta(beta)
        
        dJdT=((self.H_matirx).dot(T)-self.data.T_obs).T\
            .dot(np.linalg.inv(self.data.C_m)).dot(self.H_matirx)
        return dJdT
    
    def compute_dJdbeta(self,beta):
        dJdbeta=(beta-self.prior.mean).T.dot(self.prior.invcov)
        return dJdbeta
    
    def compute_dJdTdT(self):
        dJdTdT=(self.H_matirx).T\
            .dot(np.linalg.inv(self.data.C_m)).dot(self.H_matirx)
        return dJdTdT
        
    def compute_dJdbetadbeta(self):
        return self.prior.invcov          
    
    def compute_dRdTdbeta(self,beta):
        T=self.Solver.get_T_beta(beta)
        dRdTdbeta=np.zeros((self.N-1,self.N-1,self.N-1))
        ind1,ind2,ind3=np.diag_indices_from(dRdTdbeta)
        dRdTdbeta[ind1,ind2,ind3]=2*self.dz**2*5e-4*T**3
        return dRdTdbeta
        
    def compute_dRdTdT(self,beta):
        T=self.Solver.get_T_beta(beta)
        dRdTdT=np.zeros((self.N-1,self.N-1,self.N-1))
        ind1,ind2,ind3=np.diag_indices_from(dRdTdT)
        dRdTdT[ind1,ind2,ind3]=6*self.dz**2*5e-4*beta*T**2
        return dRdTdT
        
    #define cost function   
    def compute_J_const(self,beta):
        T=self.Solver.get_T_beta(beta)
        return 0.5*((self.H_matirx).dot(T)-self.data.T_obs).T\
            .dot(np.linalg.inv(self.data.C_m))\
                .dot((self.H_matirx).dot(T)-self.data.T_obs)\

    #define adjoint equation
    def compute_psi(self,dRdT,dJdT):
        psi=np.linalg.solve(dRdT.T,-dJdT.T)
        return psi
    
        
    def compute_gradient_adjoint_const(self,beta):
        T=self.Solver.get_T_base()
        
        dRdbeta=self.compute_dRdbeta(beta)
        #dJdbeta=self.compute_dJdbeta(beta)
        dRdT=self.compute_dRdT(beta)
        dJdT=self.compute_dJdT(beta)
        psi=self.compute_psi(dRdT,dJdT)
        
        G=psi.T.dot(dRdbeta)
        return G

    def compute_Hessian_adjoint_adjoint_const(self, beta):
        T = self.Solver.get_T_beta(beta)
        
        dRdbeta = self.compute_dRdbeta(beta)
        dRdT = self.compute_dRdT(beta)
        dJdT = self.compute_dJdT(beta)
        psi = self.compute_psi(dRdT, dJdT)
        
        dRdTdbeta = self.compute_dRdTdbeta(beta)
        dJdTdT = self.compute_dJdTdT()
        dRdTdT = self.compute_dRdTdT(beta)
        #dJdbetadbeta = self.compute_dJdbetadbeta()
        
        nu = -dRdbeta.T.dot(np.linalg.inv(dRdT))
        
        mu = np.einsum('ik,mk->im', 
                       -np.einsum('m,mik->ik', psi, dRdTdbeta)-nu.dot(dJdTdT)\
                           -np.einsum('in,m,mnk->ik', nu, psi, dRdTdT),
                       np.linalg.inv(dRdT))
        
        H = mu.dot(dRdbeta)+np.einsum('in,m,mnj->ij', nu, psi, dRdTdbeta)
        return H
    def compute_Hessian_adjoint_direct(self, beta):
        T= self.Solver.get_T_beta(beta)
        
        dRdbeta = self.compute_dRdbeta(beta)
        dRdT = self.compute_dRdT(beta)
        dJdT = self.compute_dJdT(beta)
        psi = self.compute_psi(dRdT, dJdT)
        
        dRdTdbeta = self.compute_dRdTdbeta(beta)
        dJdTdT = self.compute_dJdTdT()
        dRdTdT = self.compute_dRdTdT(beta)
        dJdbetadbeta = self.compute_dJdbetadbeta()
        
        dTdbeta = -np.linalg.inv(dRdT).dot(dRdbeta)

        temp1 = np.einsum('nk,kj->nj', dJdTdT, dTdbeta)+\
                np.einsum('m,mnj->nj', psi, dRdTdbeta)+\
                np.einsum('m,mnk,kj->nj', psi, dRdTdT, dTdbeta)
        
        dpsidbeta = -np.einsum('nj,mn->mj', temp1, np.linalg.inv(dRdT))
        
        H = dJdbetadbeta+np.einsum('m,mik,kj->ij', psi, dRdTdbeta, dTdbeta)+\
                        np.einsum('mj,mi->ij', dpsidbeta, dRdbeta)
        assert np.allclose(dJdTdT, dJdTdT.T)
        return H