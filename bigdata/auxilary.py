# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:37:51 2015

@author: ayanami
"""
import numpy as np
def set_randomseed(seed=None):
    np.random.seed(seed)
def bootstrap(n_elems,n_sample):
    return np.random.random_integers(0,n_elems-1,n_sample)
def _inthread_fit_base(estimator,X,Y,W):
    estimator.fit(X,Y,W)
    return estimator
def _inthread_predict_base(estimator,X):
    return estimator.predict(X)