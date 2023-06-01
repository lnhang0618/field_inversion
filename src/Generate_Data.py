'''
@author:lnhang

Created on Oct 15 2022
'''
import numpy as np

# import pandas as pd

# epsilon depends on Temperature
def epsilon(T,rand):
    return (1+5*np.sin(3*np.pi*T/200)+np.exp(0.02*T)+rand)*1e-4

class Generate_Data(object):
    '''
    using true epsilon to generate training data
    '''
    def __init__(self,num,h,T_inf) :
        self.num=num
        self.h=h
        self.T_inf=T_inf
        self.z_obs = np.linspace(0, 1, num+1)
        '''
        here T_inf is temperature of surroundings and 
        
        the h is Convection coefficient 
        
        num is the number of data 
        
        '''
        
        self.z=np.linspace(0,1,num+1)
        self.dz=self.z[1]-self.z[0]
        
        '''
        here we define z and dz
        '''
        
    def get_T_true(self,norm=1e-7,iter_max=80000):
        '''
        get true  information of T by using closed equation
        use finite difference method
        '''
        T=np.zeros(self.num+1)
        rand=np.random.randn(self.num-1)*0.1
        
        L2=10
        iter=0
        '''
        vectorization
        
        '''
        while L2>norm and iter<iter_max:
            T_old=T.copy() #二阶差分
            new=0.5*(T[:self.num-1]+T[2:]+\
                self.dz**2*(epsilon(T[1:self.num],rand)*\
                (self.T_inf**4-T[1:self.num]**4)+self.h*(self.T_inf-T[1:self.num])))
        
            T[1:self.num]=0.25*new+0.75*T[1:self.num]
            #T[1:self.num]=new
            L2=np.linalg.norm(T-T_old)
            iter+=1
        
        '''
        iterate this process to generate true value
        '''
        
        
        return T[1:-1]
    def run(self,C_m_type,times=100,scalar_noise=0.02):
        '''
        synthetic data 
        solving 100 realizations
        '''
        data=[]
        for i in range(times):
            tmp=self.get_T_true()
            data.append(tmp)
        self.T_true=np.vstack(data)
        #print("data:",data)
        #print("vstack:",np.vstack(data))
        self.T_obs=np.mean(self.T_true,axis=0)
        
        if C_m_type=='scalar':
            self.C_m = np.diag(scalar_noise**2*np.ones((self.num-1,)))
            
        elif C_m_type=='vector':
            #vector_noise = np.std(self.T_true, axis=0)
            vector_noise = np.mean(np.std(self.T_true, axis=0))
            self.C_m = np.diag(vector_noise**2*np.ones((self.num-1,)))
        
        
        elif C_m_type=='matrix':
            self.C_m = np.cov(self.T_true, rowvar=False)



