# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 01:26:52 2015

@author: ayanami

made in Anaconda 2.7 x64 spyder
"""
#data
from sklearn import datasets
#model
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

#eval
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

mnist = datasets.fetch_mldata("MNIST Original")
X = np.asarray(mnist.data, 'float32')/255.
Y = mnist.target.astype("int0")

Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.3,  random_state=0)



logistic = linear_model.LogisticRegression()
logistic.C = 50. #optimize me pls
logistic.penalty = 'l2'

rbm = BernoulliRBM(random_state=0, verbose=True)
rbm.n_components = 300# and me
rbm.learning_rate = 0.05
rbm.n_iter = 30
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
classifier.fit(Xtr, Ytr)



preds = classifier.predict(Xts)
print classification_report(Yts, preds)