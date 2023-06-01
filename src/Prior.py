'''
Created on Oct 17 2022

@author:lnhang
'''
import numpy as np

class Prior(object):
    def __init__(self,mean,cov) :
        self.mean=mean
        self.cov=cov
        self.invcov=np.linalg.inv(cov)
        