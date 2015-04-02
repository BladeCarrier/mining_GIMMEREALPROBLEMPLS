# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 02:30:18 2015

@author: ayanami
"""
import numpy as np
import scipy as sp
from scipy.special import expit

import copy
class _LogLoss:
    def __call__(self,factory, pred = None, margin = None):
        """
        i know it isn't; send either prediction or margin
        """
        if margin == None:
            margin = pred * factory.labels_sign
        return factory.weights *np.logaddexp(0, - margin) #np.log(1+ sp.special.expit(- margin))
    def update_leaves(self,factory,margin,tree,lrate,regularizer = 0.):
        '''
        update leaf values via... Newton guy...
        '''

        leaf_indices = factory.getLeafIndices(tree)
        leaf_values = tree[2]*0
        normalizers = np.zeros(leaf_values.shape[0])

        expt = expit(-margin)
        prec_value = factory.weights*factory.labels_sign*expt
        prec_norm = (expt) * (1 - expt)*factory.weights
        
        count_v = np.bincount(leaf_indices, weights=prec_value, minlength=64)
        count_n = np.bincount(leaf_indices, weights=prec_norm, minlength=64)
        
        leaf_values = count_v[:len(leaf_values)]
        normalizers = count_n[:len(leaf_values)]+regularizer
        leaf_values[normalizers !=0] /= normalizers[normalizers !=0]

        newtree = tuple([copy.copy(i) for i in tree[:2]] + [leaf_values*lrate])

        return newtree
LogLoss = _LogLoss()

def entropy(distribution):
    """just some entropy"""
    logs = np.array(map(np.log,distribution))
    logs[distribution ==0] = 0
    return -sum(distribution*logs)