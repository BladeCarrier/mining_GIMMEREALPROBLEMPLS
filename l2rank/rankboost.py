# -*- coding: utf-8 -*-


#i had to rewrite every freaking line here, reference no longer needed

import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn.tree import DecisionTreeRegressor
import sys

class RankBoost(BaseEnsemble):
    """generic non-bipartite ranker as described in freund3a:http://jmlr.csail.mit.edu/papers/volume4/freund03a/freund03a.pdf"""
 
    def __init__(self,
                 n_estimators=50,
                 learning_rate=1.,
                 verbose=0,
                 base_estimator = None):
        if base_estimator==None:
            base_estimator=DecisionTreeRegressor()

        super(RankBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=tuple())
        self.base_estimator_ = base_estimator
        self.estimator_weights_ = None
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.scale = 1.
        self.shift = 0.        

    def fit(self, X, y):
        
        self.scale = max(y)-min(y)
        self.shift = min(y)
        
        y= (y-self.shift) / self.scale
        
        X, y = np.array(X), np.array(y)
        

        # first sort on y, 4-3-2-1-0 order
        sortidx = y.argsort()[::-1]
        X, y = X[sortidx], y[sortidx]
        
        score_types = list(set(y))
        score_types.sort(reverse=True)
        
        counts= {i: sum(y == i) for i in score_types}
                
        # equalize weights for all the rankings
        sample_weight = np.empty(X.shape[0], dtype=np.float)
        prefix_sum=0
        for score in score_types:
            sample_weight[prefix_sum:prefix_sum+counts[score]] = 1./counts[score]
            prefix_sum += counts[score]
        
        self.estimators_ = []
        self.estimator_weights_ =  np.zeros(self.n_estimators, dtype=np.float)
        
        # Create argsorted X for fast tree induction
        X_argsorted = np.asfortranarray(np.argsort(X.T, axis=1).astype(np.int32).T)
                
        for iboost in xrange(self.n_estimators):
            sample_weight, estimator_weight = self._boost(iboost,
                                                          X, y,
                                                          sample_weight,
                                                          X_argsorted=X_argsorted,
                                                          counts=counts)
    
            self.estimator_weights_[iboost] = estimator_weight
            
            prefix_sum=0
            for score in score_types:
                sample_weight[prefix_sum:prefix_sum+counts[score]] /= np.sum(sample_weight[prefix_sum:prefix_sum+counts[score]])
                prefix_sum += counts[score]        
            
        return self
    
    def _boost(self, iboost, X, y, sample_weight, X_argsorted, counts):
        
        estimator = self._make_estimator()

        if self.verbose == 2:
            print 'building tree', iboost+1, 'out of', self.n_estimators
        elif self.verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        
        sortidx = y.argsort()[::-1]
        X, y = X[sortidx], y[sortidx]
        estimator.fit(X, y, sample_weight=sample_weight)


        
        y_predict = estimator.predict(X)

        score_types = list(set(y))
        score_types.sort(reverse=True)
        
        #freund3a:http://jmlr.csail.mit.edu/papers/volume4/freund03a/freund03a.pdf
        #third and last suggested way of chosing weight for any real a3        
        #why that: 
        #1) a = 1/2 ln((1+r)/(1-r)) where
        # r = sum[over x0,x1 in X](  D(x0,x1)*(h(x1)-h(x0))  )*W0*W1
        # where D(x0,x1) ~= y1-y0   ##at leat i thought so...
        
        # sum[over x0,1 in X:y(x) = y0,1] ( (h(x1) - h(x0) )*w0*w1 ) =
        # = sum[x1] ( h(x1)*w1 )*sum(w0) - sum[x0] ( h(x0)*w0) *sum(w1)
        
        # thus, r = sum[yi,yj, j!=i]( (yj-yi)*(sum_Hj_wj*sum_wi - sum_Hi_wi*sum_wj) =
        # = sum[yi,yj, j>i]( 2*(yj-yi)*(sum_Hj_wj*sum_wi - sum_Hi_wi*sum_wj) )
        # where sum_Hk_wk = sum[i:yi = k] ( H(xi)*wi )
        # sum_wk = sum[i:yi = k](wi)
        #which can both be precomputed to grant O(n_y^2 * N) complexity 
        #where n_y is a number of distinct Ys and N is training sample length
        
        #precompute sum_wk,sum_Hk_wk
        #in case we renormalize all wk after every iteration, all sum_wk == 1.
        sum_w = {}
        sum_h_w = {}
        prefix_sum=0
        for score in score_types:
            sum_w[score] = sum(sample_weight[prefix_sum:prefix_sum+counts[score]])
            sum_h_w[score] = sum(sample_weight[prefix_sum:prefix_sum+counts[score]]*y_predict[prefix_sum:prefix_sum+counts[score]])
            prefix_sum += counts[score]        
        

        err = 0.
        for i in range(len(score_types)):
            for j in range(i+1,len(score_types)):
                yi = score_types[i]
                yj = score_types[j]
                err += 2*(yj-yi)*(sum_h_w[yj]*sum_w[yi]-sum_h_w[yi]*sum_w[yj])
                
        err /=  sum(sample_weight)
                
        estimator_weight = self.learning_rate*np.log((1.+err)/(1.-err))/2.
        
        prefix_sum=0
        for score in score_types:
            sample_weight[prefix_sum:prefix_sum+counts[score]] *=np.exp(-estimator_weight*y_predict[prefix_sum:prefix_sum+counts[score]])
            prefix_sum += counts[score]        
        
                
        return sample_weight, estimator_weight
    def predict(self,X):
        X = np.array(X)
        
        y_predict = np.zeros(X.shape[0])

        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            y_predict += estimator.predict(X)*weight        
        return y_predict*self.scale + self.shift
    def staged_predict(self,X):
        
        X = np.array(X)
        
        y_predict = np.zeros(X.shape[0])

        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            y_predict += estimator.predict(X)*weight        
            yield y_predict*self.scale + self.shift
        
    
        