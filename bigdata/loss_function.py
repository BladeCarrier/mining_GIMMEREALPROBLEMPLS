# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:05:28 2015

@author: ayanami
"""
from sklearn.ensemble.gradient_boosting import BinomialDeviance
import numpy as np
class logloss(BinomialDeviance):
    def __init__(self):
        BinomialDeviance.__init__(self,2)
    def update_leaves(self,rf,X,Y,W,Ypred,residual,lrate  =1., k =1):
        return
        sample_mask = sample_mask = np.ones((len(Y), ), dtype=np.bool)
        Y = Y.reshape((len(Y),1))
        Ypred = np.concatenate([-Ypred,Ypred]).reshape(2,len(Ypred)).transpose()
        for tree in rf.estimators:
            self.update_terminal_regions(tree.tree_,
                                         X, Y, 
                                         residual,
                                         Ypred,
                                         W, sample_mask,
                                         lrate, k=k)
    def residual(self,Y,Ypred,W,k=1):
        return self.negative_gradient(Y, Ypred, k=k, sample_weight=W)
