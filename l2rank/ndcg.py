# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:07:02 2015

@author: ayanami
"""
import numpy as np
def dcg(relevances, rank=None):
    """Discounted cumulative gain"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)
 
 
def ndcg(relevances, rank=None):
    """Normalized DGC"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg
def mean_ndcg(y_true, y_pred, query_ids, rank=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    query_ids = np.asarray(query_ids)
    # assume query_ids are sorted
    ndcg_scores = []
    previous_qid = query_ids[0]
    previous_loc = 0
    for loc, qid in enumerate(query_ids):
        if previous_qid != qid:
            chunk = slice(previous_loc, loc)
            ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
            ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
            previous_loc = loc
        previous_qid = qid

    chunk = slice(previous_loc, loc + 1)
    ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
    ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
    return np.mean(ndcg_scores)
