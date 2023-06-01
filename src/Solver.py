import numpy as np
'''
Created on Oct 16 2022

@author:lnhang
'''

class Solver(object):
    def __init__(self,num,h,T_inf):
        self.num=num
        self.h=h
        self.T_inf=T_inf
        '''
        here T_inf is temperature of surroundings and 
        
        the h is Convection coefficient 
        
        num is the number of data 
        
        '''
        
        self.z=np.linspace(0,1,num+1)
        self.dz=self.z[1]-self.z[0]
    
    def get_T_base(self,norm=1e-7,iter_max=80000):
 
        T=np.zeros(self.num+1)
        
        L2=10
        iter=0
        '''
        vectorization
        
        '''
        while L2>norm and iter<iter_max:
            T_old=T.copy() #二阶差分
            new=0.5*(T[:self.num-1]+T[2:]+\
                self.dz**2*(5e-4)*\
                (self.T_inf**4-T[1:self.num]**4))
        
            T[1:self.num]=0.5*new+0.5*T[1:self.num]
            L2=np.linalg.norm(T-T_old)
            iter+=1
        
        '''
        iterate this process to generate approximate value
        '''
        
        return T[1:-1]
    
    def get_T_beta(self,beta,norm=1e-5,iter_max=80000):
 
        T=np.zeros(self.num+1)
        
        L2=10
        iter=0
        '''
        vectorization
        
        '''
        while L2>norm and iter<iter_max:
            T_old=T.copy() #二阶差分
            new=0.5*(T[:self.num-1]+T[2:]+\
                self.dz**2*(5e-4)*beta*\
                (self.T_inf**4-T[1:self.num]**4))
        
            T[1:self.num]=0.25*new+0.75*T[1:self.num]
            L2=np.linalg.norm(T-T_old)
            iter+=1
        
        '''
        iterate this process to generate approximate value
        '''
        
        return T[1:-1]
    
    def get_beta_r_true(self, T, with_rand=True):
        if with_rand:
            rand = np.random.normal(loc=0, scale=0.1, size=(T.shape))
        else:
            rand = 0
        return (1/5e-4)*(1+5*np.sin(3*np.pi*T/200)+np.exp(0.02*T)+rand)*1e-4
        
        
    def get_beta_c_true(self, T):
        return self.h/5e-4*(self.T_inf-T)/(self.T_inf**4-T**4)
    
    def get_beta_true(self, T,with_rand=True):

        
        beta_r = self.get_beta_r_true(T,with_rand)
        beta_c = self.get_beta_c_true(T)
        return beta_r+beta_c

    def get_T_ML(self,ML_model,norm=1e-7,iter_max=80000):

        T=np.zeros(self.num+1)

        L2=10

        iter = 0

        while L2 > norm and iter < iter_max:
            if iter % 100 == 0:
                print(iter)
            T_old = T.copy()

            beta = np.ones((self.num - 1))
            for i in range(1, self.num):
                beta[i - 1] = ML_model.predict(np.array([T[i], self.T_inf[i - 1]]).reshape(1, -1))[0]

            new = 0.5 * (T[:self.num - 1] + T[2:] - \
                         self.dz ** 2 * 5e-4 * beta * (T[1:self.num] ** 4 - self.T_inf ** 4))

            alpha = 0.5
            T[1:self.num] = alpha * new + (1 - alpha) * T[1:self.num]


            iter += 1


        return T[1:-1]