'''
Created on Nov 5 2022

@author:lnhang

'''

import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from scipy.optimize import minimize
'''
class Dataset:

    def __init__(self,path_train,path_test):
        self.path_train=path_train
        self.path_test=path_test

        self.x_train,self.y_train=self.load_data_train()
        self.y_train,self.y_test=self.load_data_test()

        self.x_train, self.x_val, self.y_train, self.y_val = self.split()

    def load_data_train(self):
        (T_inf_train,
         T_obs_train,
         beta_MAP_train,
         std_lst_train) = pickle.load(open(self.path_train, 'rb'))

        T_inf_train = np.vstack(T_inf_train).flatten()
        T_obs_train = np.vstack(T_obs_train).flatten()
        beta_MAP_train = np.vstack(beta_MAP_train).flatten()
        std_lst_train = np.vstack(std_lst_train).flatten()

        x_train = np.concatenate([T_obs_train[:, np.newaxis], T_inf_train[:, np.newaxis]],
                                 axis=1)
        y_train = beta_MAP_train
        return x_train, y_train,std_lst_train

    def load_data_test(self):
        (T_inf_test,
         T_true_test,
         beta_true_test) = pickle.load(open(self.path_test, 'rb'))

        T_inf_test = np.vstack(T_inf_test).flatten()
        T_true_test = np.vstack(T_true_test).flatten()
        beta_true_test = np.vstack(beta_true_test).flatten()

        x_test = np.concatenate([T_true_test[:, np.newaxis],
                                 T_inf_test[:, np.newaxis]], axis=1)
        y_test = beta_true_test
        return x_test, y_test
'''
class Guassian_Process():

    def __init__(self,sigma_noise,optimize=True):
        self.is_fit=False
        self.optimize=optimize
        self.h=20
        self.sigma_noise=sigma_noise

    def train(self,x_train,y_train):

        '''
        x_train is (31,20)

        y_train 15 (31,10)

        '''

        N = len(x_train)

        self.x_train=x_train
        self.y_train=y_train

        def likehood_fuc(h):
            self.h=h
            N=len(self.y_train)
            K = self.compute_K(self.x_train, self.x_train)
            K_y = K + self.sigma_noise ** 2 * np.eye(N)
            res = 0.5 * np.linalg.slogdet(K_y)[1] + 0.5 * y_train.T.dot(np.linalg.inv(K_y)).dot(y_train) + N*0.5\
                        * np.log(2 * np.pi)

            return res.ravel()
        if self.optimize:
            h_solved = minimize(likehood_fuc,self.h,options={"disp":True,"maxiter":1000})
            self.h = h_solved.x
        self.is_fit=True

        self.K = self.compute_K(x_train,x_train)
        self.K_y = self.K+self.sigma_noise**2*np.eye(N)+1e-8 * np.eye(N)
        self.K_yinv=np.linalg.inv(self.K_y)

    def compute_K(self,x,xp):
        N=len(x)
        Np=len(xp)

        K=np.ones((N,Np))
        for i in range(N):
            for j in range(Np):
                K[i][j]=self.kernel(x[i,:],xp[j,:])
        return K

    def predict(self,x_test):

        self.K_fy=self.compute_K(self.x_train,x_test)
        self.K_yy=self.compute_K(x_test,x_test)+self.sigma_noise**2*np.eye(len(x_test))

        mu=self.K_fy.T.dot(self.K_yinv).dot(self.y_train)
        sigma=self.K_yy-self.K_fy.T.dot(self.K_yinv).dot(self.K_fy)

        return mu,sigma

    def kernel(self,x,xp):
        return np.exp(-np.linalg.norm(x - xp) ** 2 / self.h ** 2)







