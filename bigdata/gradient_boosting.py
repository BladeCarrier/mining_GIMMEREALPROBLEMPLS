# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:37:41 2015

@author: ayanami
"""


import numpy as np
from scipy.special import expit
from loss_function import logloss
from random_forest import RandomForestBuilder
class GradientBoostingBuilder:
    def __init__(self,
                 loss_function = 'logloss',
                 n_iterations=10,
                 learning_rate =0.1,
                 batch_size = 10000,
                 iterations_per_batch=1,
                 rf_params={}):
        self.batch_size = batch_size
        self.n_iterations= n_iterations
        self.learning_rate=learning_rate
        self.loss_function = loss_function
        if self.loss_function =='logloss':
            self.loss_function = logloss()
        self.iterations_per_batch = iterations_per_batch
        self.rf_params = rf_params
        self.estimators = []
    def predict(self,X):
        pred = np.zeros(len(X))
        for esti in self.estimators:
            pred += esti.predict(X)
        return pred
    def predict_proba(self,X):
        pred = np.zeros(len(X))
        for esti in self.estimators:
            pred += esti.predict(X)
        p = expit(pred)
        return p
        #return np.concatenate([1-p,p]).reshape(2,len(p)).transpose()

        
        
        
    def fit(self,stream,verbose = 0):                
        for i in xrange(self.n_iterations):
            X,Y,W = stream.readSplitted(self.batch_size)
            
            for e in xrange(self.iterations_per_batch):
                Ypred = self.predict_proba(X)
                residual = self.loss_function.residual(Y,Ypred,W)*self.learning_rate

                esti  = RandomForestBuilder(**self.rf_params).fit(X,residual,W)

                self.loss_function.update_leaves(esti,
                                                 X,Y,W,
                                                 Ypred,residual,
                                                 lrate  =self.learning_rate) 
                
                self.estimators.append(esti)
                if verbose:
                    print "\niteration #",(i*self.iterations_per_batch)+e+1
                    print "loss",self.loss_function(Y,Ypred,W)

        return self
                
            
            
            
            
            
        
        