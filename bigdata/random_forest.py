# -*- coding: utf-8 -*-
"""
RandomForest for gradient boosting
Created on Tue Apr 14 17:33:47 2015
@author: ayanami
"""
import copy
import numpy as np
from sklearn.externals import joblib as jl
from sklearn.tree import DecisionTreeRegressor
tree_sklearn = DecisionTreeRegressor

from auxilary import bootstrap,_inthread_fit_base

class RandomForestBuilder:
    def __init__(self,n_estimators=100,
                 subsample_size = 1.,
                 use_joblib=True,n_jobs=-1,
                 tree_class = None, tree_params={}):
        self.use_joblib = True
        self.n_jobs =-1
        self.subsample_size = subsample_size
        self.tree_class = (tree_class if tree_class!= None else tree_sklearn)
        self.tree_params = tree_params
        self.n_estimators=n_estimators
        self.estimators =[]
        
    def fit(self,X,Y,W):
        subsample= self.subsample_size if self.subsample_size>1 else (self.subsample_size*len(Y))
        subsample= int(subsample)
        tree = self.tree_class(**self.tree_params)
        subsample_ind = [bootstrap(len(Y), subsample) for i in xrange(self.n_estimators)]
        
        if self.use_joblib:
            jobs = [jl.delayed(_inthread_fit_base)(
                                copy.deepcopy(tree),
                                X[subsample_ind[i]],
                                Y[subsample_ind[i]],
                                W[subsample_ind[i]])
                                for i in xrange(self.n_estimators)]
                                    
            self.estimators = jl.Parallel(n_jobs=self.n_jobs,backend="threading")(jobs)
        else:
            self.estimators = [_inthread_fit_base(
                                    copy.deepcopy(tree),
                                    X[subsample_ind[i]],
                                    Y[subsample_ind[i]],
                                    W[subsample_ind[i]])
                                for i in xrange(self.n_estimators)]
        return self
    def predict(self,X):
        ans = np.zeros(len(X))
        for tree in self.estimators:
            ans += tree.predict(X)
        return ans


#treeparams:
#            tree = DecisionTreeRegressor(
#                criterion=criterion,
#                splitter=splitter,
#                max_depth=self.max_depth,
#                min_samples_split=self.min_samples_split,
#                min_samples_leaf=self.min_samples_leaf,
#                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
#                max_features=self.max_features,
#                max_leaf_nodes=self.max_leaf_nodes,
#                random_state=random_state)