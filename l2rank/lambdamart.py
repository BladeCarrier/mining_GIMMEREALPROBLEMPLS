#The code was written in a spree of hatred

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn.base import BaseEstimator
from ndcg import mean_ndcg
class LambdaMART(BaseEstimator):
    def __init__(self,n_estimators=100, learning_rate=0.1,max_depth = 6, k=10):
        super(LambdaMART, self).__init__()
        self.estimators_= []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.rank = k
        self.max_depth =max_depth

    def __len__(self):
        return len(self.estimators_)


    def predict(self, X):
        results = np.zeros(len(X))
        for tree in self.estimators_:
            results += tree.predict(X) * self.learning_rate
        return results
    def staged_predict(self,X):
        results = np.zeros(len(X))
        for tree in self.estimators_:
            results += tree.predict(X) * self.learning_rate
            yield results
        

        
    def fit(self,X,Q,Y,Xval=None,Qval=None,Yval=None):
    
            
            
        
        print "Training starts..."
        Y_pred = np.array([float(0)] * len(X))
        if Xval != None:
            Yval_pred = np.array([float(0)] * len(Xval))
    
        for i in range(self.n_estimators):
            print " Iteration: " + str(i + 1)
            # Compute lambdas
            lambdas = Y-Y_pred
            # create tree and append it to the model
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, lambdas)
    
            self.estimators_.append(tree)
    
            # update model score
            prediction = tree.predict(X)
            Y_pred += self.learning_rate * prediction
    
            # train_score
            train_score = mean_ndcg(Y,Y_pred, Q,self.rank)
            print "train score " + str(train_score)
    
            # validation score, in case a validation set is provided.
            if Yval != None:
                Yval_pred += self.learning_rate * tree.predict(Xval)
                val_score = mean_ndcg( Yval,Yval_pred, Qval, self.rank)
                print "validation score " + str(val_score)
    
    
    
        print "training finished."
        return self














