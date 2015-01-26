# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 22:27:36 2015
#что это: это куски кода, собранные так, что при их исполнении получится готовая модель,
#которая заклассифаит тесты и напишет их в CSV
#Требуется: training.csv, test.csv в рабочей директории
#Output: higgs.pred.csv в рабочей директории, готовый к kaggle-посылке.

@отмазка: код делался за сутки с учётом сна, еды и сданного экзамена. К стуктуре прошу не придираться: её нет.
@author: ayanami
"""


import csv
import math
import os

import numpy as np

from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.tree import *


from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.grid_search import *

#os.system('taskset -p 0xffffffff %d' % os.getpid())


print 'Loading training data.'
data = np.loadtxt('training.csv', \
        delimiter=',', \
        skiprows=1, \
        converters={32: lambda x:int(x=='s'.encode('utf-8'))})

X = data[:,1:31]
Y = data[:,32]
W = data[:,31]

print 'Loading testing data.'
testdata = np.loadtxt('test.csv', \
    delimiter=',', \
    skiprows=1)

idstest = testdata[:,0]
Xtest = testdata[:,1:31]
W = data[:,31]

print "Learning..."
print "doing preproc"
# илея, вкратце - забить пропуски в данных и перевести положительные переменные в логарифмическую шкалу.
#last str: mean

imputer = Imputer(missing_values = -999.0, strategy = 'most_frequent')
X = imputer.fit_transform(X)
Xtest = imputer.transform(Xtest)

indTransform = (0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26)
Xtrans = np.log(1 / (1 + X[:, indTransform]))
X = np.hstack((X, Xtrans))
XtransTest = np.log(1 / (1 + X_test[:, indTransform]))
Xts = np.hstack((Xts, XtransTest))

# подогнать вариансы.
scaler = StandardScaler()
X = scaler.fit_transform(X)
Xts = scaler.transform(Xts)
print "model init"
#last ntrs 150

#почему?
#1) деревья круто бустятся
#2) адабуст быстрее и имхо не хуже xg (см. R-sln)
#3) леса на нижнем уровне = быстрый параллельный бэггинг нахаляву. Все плюшки прилагаются. Эмпирически круче простого буста
#4) параметры подсказал Ктулху и форум.ещё раз тренить 6000 деревьев я не буду.
rf = ExtraTreesClassifier(
            n_estimators = 300,
            max_features = 30,
            max_depth = 12,
            min_samples_leaf = 100,
            min_samples_split = 100,
            verbose = 1,
            n_jobs = -1)
classifier = AdaBoostClassifier(
        n_estimators = 20,
        learning_rate = 0.75,
        base_estimator = rf)
print "model fit"

classifier.fit(X, Y, sample_weight = W)
print "predict"
Ypred = classifier.predict_proba(X)[:,1]
Yts_pred = classifier.predict_proba(Xts)[:,1]

#last:15
signal_threshold = 84
cut = np.percentile(Yts_pred, signal_threshold)
Ysig = Ypred > cut
Yts_sig = Yts_pred > cut

print "Saving results"
#пишем csv
idsProbs = np.transpose(np.vstack((idsTest, Yts_pred)))
idsProbs = np.array(sorted(idsProbs, key = lambda x: -x[1]))
idsRanks = np.hstack((
    idsProbs,
    np.arange(1, idsProbs.shape[0]+1).reshape((idsProbs.shape[0], 1))))

idsMap = {}
for tsid, prob, rank in idsRanks:
    tsid = int(tsid)
    rank = int(rank)
    idsMap[tsid] = rank

f = open('higgs.pred.csv', 'wb')
writer = csv.writer(f)
writer.writerow(['EventId', 'RankOrder', 'Class'])
for i, pred in enumerate(Yts_sig):
    event_id = int(idstest[i])
    rank = idsMap[idstest[i]]
    klass = pred and 's' or 'b'
    writer.writerow([event_id, rank, klass])
f.close()



def ams(s, b):
    return math.sqrt(2 * ((s + b + 10) * math.log(1.0 + s/(b + 10)) - s))
