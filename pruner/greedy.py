# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 02:30:48 2015

@author: ayanami
"""

from sklearn.externals import joblib
import numpy as np
import scipy as sp
import random,copy

def inthread_try_add(bunch,tree,factory,loss,margin,y_pred,learning_rate,regularizer):
        '''in case of joblibification, use this (c)'''
        newTree = loss.update_leaves(factory,margin,tree,learning_rate,regularizer)
        newPred = y_pred + factory.predict([newTree])
        newLoss = np.sum(loss(factory,newPred))
        return newLoss,newTree,newPred

def try_add1_bfs(bunch, allTrees,factory,learning_rate,loss,breadth,y_pred = None,regularizer = 0.):
    '''
    select best tree to add (1 step)
    '''
    y_sign = factory.labels_sign
    if y_pred == None:
        y_pred = factory.predict(bunch)
    margin = y_sign*y_pred
    triples = [inthread_try_add(bunch,tree,factory,loss,margin,y_pred,learning_rate,regularizer) for tree in allTrees]   
    triples.sort(key = lambda el: el[0])
    
    bunches = []
    preds = []
    for triple in triples[:breadth]:
        tree = triple[1]
        pred = triple[2]
        bunch = bunch+[tree]
        bunches.append(bunch)
        preds.append(pred)

    return bunches,[triple[1] for triple in triples[:breadth]],[triple[0] for triple in triples[:breadth]],preds


def greed_up_features_bfs (trees,
                           factory,
                           loss,
                           learning_rate,
                           breadth,
                           nTrees,
                           trees_sample_size,
                           verbose = True,
                           learning_rate_decay = 1.,
                           trees_sample_increase = 0,
                           regularizer = 0.):
    """
    Iterative BFS over best ADD-1 results for [nTrees] iterations
    """
    allTrees = copy.copy(trees)
    
    trees_sample = np.array(random.sample(allTrees,trees_sample_size))
    
    bunches,additions,losses,preds = try_add1_bfs([],trees_sample,factory,learning_rate,loss,breadth,regularizer = regularizer)
    bestScore = min(losses)

    if verbose:
        print "\niteration #",0," ntrees = ", len(bunches[0]),"\nbest loss = ",bestScore
        print "learning_rate = ", learning_rate
        print "sample_size", trees_sample_size

    
    itr = 0
    while len(bunches[0]) <nTrees:

        itr+=1
        newBunches = []    
        newScores = []
        newPreds = []
        for bunch,pred in zip(bunches,preds):
            trees_sample = np.array(random.sample(allTrees,trees_sample_size))
            _bunches,_additions,_losses,_preds = try_add1_bfs(bunch,trees_sample,factory,learning_rate,loss,
                                                              breadth,pred,regularizer=regularizer)
            newBunches+=_bunches
            newScores += _losses
            newPreds += _preds
            
        learning_rate *= learning_rate_decay
        trees_sample_size = min(len(allTrees),trees_sample_size + trees_sample_increase)
            
        triples = zip(newScores,newBunches,newPreds)
        triples.sort(key = lambda el: el[0])
        
        
        newBestScore = min(newScores)
        
        if newBestScore > bestScore:
            learning_rate /=2.
            if learning_rate < 0.00001:
                break
        else: 
            bestScore = newBestScore
            bunches = [triple[1] for triple in triples[:breadth]]       
            preds = [triple[2] for triple in triples[:breadth]]       

        
        
        if verbose:
            print "\niteration #",itr," ntrees = ", len(bunches[0]),"\nbest loss = ", bestScore,"\nlast loss = ",newBestScore
            print "learning_rate = ", learning_rate
            print "sample_size", trees_sample_size          
    return bunches[0]


def wheel_up_features_bfs (initialBunch,
                           trees,
                           factory,
                           loss,
                           learning_rate,
                           nIters,
                           trees_sample_size,
                           verbose = True,
                           learning_rate_decay = 1.,
                           trees_sample_increase = 0,
                           regularizer = 0.,
                           random_walk = True):
    """
    Iterative BFS over best ADD-1 results for [nTrees] iterations
    """
    allTrees = copy.copy(trees)
    
    bunch = copy.copy(initialBunch)
    pred = factory.predict(bunch)
    bestScore = sum(loss(factory,pred))

    if verbose:
        print "\niteration #",0," ntrees = ", len(bunch),"\nbest loss = ",bestScore
        print "learning_rate = ", learning_rate
        print "sample_size", trees_sample_size

    
    for i in xrange(1,nIters+1):
        change_index= random.randint(0,len(bunch)-1) if random_walk else  (i-1)%len(bunch)
        trees_sample = random.sample(allTrees,trees_sample_size)+ [bunch[change_index]]
        bunch_wo = copy.copy(bunch)
        bunch_wo.pop(change_index)
        newBunches,_,newScores,newPreds = try_add1_bfs(bunch_wo,
                                                     trees_sample,
                                                     factory,
                                                     learning_rate,
                                                     loss,
                                                     1,
                                                     None,#pred - factory.predict([bunch[change_index]]),
                                                     regularizer=regularizer)
        
        learning_rate *= learning_rate_decay
        trees_sample_size = min(len(allTrees),trees_sample_size + trees_sample_increase)
            
        triples = zip(newScores,newBunches,newPreds)
        triples.sort(key = lambda el: el[0])
        newBestScore = min(newScores)
        
        if newBestScore > bestScore:
            pass
        else: 
            bestScore = newBestScore
            bunch = triples[0][1]
            bunch.insert(change_index,bunch.pop())
            pred = triples[0][2]

        
        
        if verbose:
            print "\niteration #",i," ntrees = ", len(bunch),"\nbest loss = ", bestScore,"\nlast loss = ",newBestScore
            print "changed index",change_index
            print "learning_rate = ", learning_rate
            print "sample_size", trees_sample_size          
    return bunch

def predict(factory,trees):
    return factory.predict(trees)