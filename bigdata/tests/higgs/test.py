# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:35:31 2015

@author: ayanami
"""

import sys
sys.path.append("../..")#path_to_library
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

#data =pd.read_csv("training.csv")
#train = data[:100000]
#test = data[100000:]
#train.to_csv("train")
#test.to_csv("test")
#del data, train,test

def data_transformer(frame):
    X = np.array(frame.icol(range(1,31)),dtype= np.float32)
    Y = np.array(frame["Label"]=='s')*2-1
    W = np.array(frame["Weight"])
    return X,Y,W

from stream import csvReader
from gradient_boosting import GradientBoostingBuilder
reader = csvReader('train',cyclic = True,
                          transformer = data_transformer)

rf_params = {'n_estimators' : 100}
model = GradientBoostingBuilder(n_iterations = 10,
                                learning_rate = .1,
                                rf_params = rf_params,
                                batch_size =10000)

model.fit(reader,1)

test =pd.read_csv('test')
Xts,Yts,Wts = data_transformer(test)
Ypr = model.predict(Xts)
print 'auc',roc_auc_score(Yts,Ypr,sample_weight= Wts)


