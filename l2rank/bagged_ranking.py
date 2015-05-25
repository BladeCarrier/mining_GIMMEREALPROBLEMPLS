# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:55:02 2015

@author: ayanami
"""
import numpy as np
import numbers
import itertools
from sklearn.ensemble import BaggingRegressor
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.random import sample_without_replacement
from sklearn.externals.joblib import Parallel,delayed
from sklearn.utils.fixes import bincount
from sklearn.ensemble.base import _partition_estimators
MAX_INT = np.iinfo(np.int32).max
from sklearn.utils import  check_random_state,check_X_y


def _parallel_build_ranking_estimators(n_estimators, ensemble, X, y, Q, sample_weight,
                               seeds, verbose):
    """Private function used to build a batch of estimators within a job.
    Now it supports queries and querywise sampling.
    It also breaks the PEP8 line length constraint now"""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_samples = ensemble.max_samples
    max_features = ensemble.max_features
    uQueries = np.unique(Q)

    sample_whole_queries = False
    if hasattr(ensemble,"sample_whole_queries"):
        sample_whole_queries = ensemble.sample_whole_queries


    if (not isinstance(max_samples, (numbers.Integral, np.integer)) and
            (0.0 < max_samples <= 1.0)):
        if sample_whole_queries:
            max_samples = int(max_samples * len(uQueries))
        else:
            max_samples = int(max_samples * n_samples)
        

    if (not isinstance(max_features, (numbers.Integral, np.integer)) and
            (0.0 < max_features <= 1.0)):
        max_features = int(max_features * n_features)

    bootstrap = ensemble.bootstrap
    
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")

    # Build estimators
    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("building estimator %d of %d" % (i + 1, n_estimators))

        random_state = check_random_state(seeds[i])
        seed = check_random_state(random_state.randint(MAX_INT))
        estimator = ensemble._make_estimator(append=False)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=seed)
        except ValueError:
            pass

        # Draw features
        if bootstrap_features:
            features = random_state.randint(0, n_features, max_features)
        else:
            features = sample_without_replacement(n_features,
                                                  max_features,
                                                  random_state=random_state)
                                                  

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                if sample_whole_queries:
                    Qindices = uQueries[random_state.randint(0, 
                                                             len(uQueries),
                                                             max_samples)]
                    Qindices.sort()
                    indices = reduce(np.append,[np.where(Q==i) for i in Qindices])

                else:                
                    indices = random_state.randint(0, n_samples, max_samples)
                sample_counts = bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts

            else:
                if sample_whole_queries:
                    notQindices = uQueries[random_state.randint(0, 
                                                                len(uQueries),
                                                                len(uQueries)-max_samples)]
                    notQindices.sort()
                    not_indices = reduce(np.append,[np.where(Q==i) for i in Qindices])
                else:
                    not_indices = sample_without_replacement(
                        n_samples,
                        n_samples - max_samples,
                        random_state=random_state)

                curr_sample_weight[not_indices] = 0

            estimator.fit(X[:, features], y, Q = Q,sample_weight=curr_sample_weight)
            samples = curr_sample_weight > 0.

        # Draw samples, using a mask, and then fit
        else:
            if bootstrap:
                if sample_whole_queries:
                    Qindices = uQueries[random_state.randint(0, 
                                                             len(uQueries), 
                                                             max_samples)]
                    Qindices.sort()
                    indices = reduce(np.append,[np.where(Q==i) for i in Qindices])

                else:                
                    indices = random_state.randint(0, n_samples, max_samples)
            else:
                if sample_whole_queries:
                    Qindices = uQueries[sample_without_replacement(
                                                     len(uQueries),
                                                     max_samples,
                                                     random_state=random_state)
                                        ]
                    Qindices.sort()
                    indices = reduce(np.append,[np.where(Q==i) for i in Qindices])

                else:                
                    indices = sample_without_replacement(n_samples,
                                                         max_samples,
                                                         random_state=random_state)

            sample_counts = bincount(indices, minlength=n_samples)

            estimator.fit((X[indices])[:, features], y[indices],Q=Q[indices])
            samples = sample_counts > 0.

        estimators.append(estimator)
        estimators_samples.append(samples)
        estimators_features.append(features)

    return estimators, estimators_samples, estimators_features
    
class BaggingRanker(BaggingRegressor):
    """I am a ranking estimator that depends on queries(Q)"""
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 sample_whole_queries = False,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(BaggingRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.sample_whole_queries = sample_whole_queries
    def fit(self, X, y, Q = None,sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])

        # Remap output
        n_samples, self.n_features_ = X.shape
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        # Free allocated memory, if any
        self.estimators_ = None

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_ranking_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                Q,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_samples_ = list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(
            t[2] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self    
    