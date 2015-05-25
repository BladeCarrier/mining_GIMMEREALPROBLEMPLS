#The code is mostly a reshuffle of various parts of github.com/jwkvam/scikit-learn

import numpy as np
import numbers
from sklearn.tree import DecisionTreeRegressor

from abc import abstractmethod, ABCMeta
from warnings import warn


from sklearn.ensemble.gradient_boosting import BaseGradientBoosting

from sklearn.externals import six
from sklearn.ensemble.gradient_boosting import LOSS_FUNCTIONS,\
            MultinomialDeviance, BinomialDeviance,\
            INIT_ESTIMATORS,VerboseReporter
from sklearn.base import BaseEstimator
from sklearn.utils import  check_random_state,check_X_y,column_or_1d

#cython section
from lmart_aux import _ranked_random_sample_mask,_ndcg,_lambda
from sklearn.tree._tree import DTYPE,TREE_LEAF,PresortBestSplitter,FriedmanMSE
from sklearn.ensemble._gradient_boosting import _random_sample_mask

#bagging section



class ZeroEstimator(BaseEstimator):
    """An estimator that simply predicts zero. """
    def __init__(self, is_classification=True):
        self.is_classification = is_classification

    def fit(self, X, y):
        if np.issubdtype(y.dtype, int) and self.is_classification:
            # classification
            self.n_classes = np.unique(y).shape[0]
            if self.n_classes == 2:
                self.n_classes = 1
        else:
            # regression
            self.n_classes = 1

    def predict(self, X):
        y = np.empty((X.shape[0], self.n_classes), dtype=np.float64)
        y.fill(0.0)
        return y
    
class LossFunction(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for various loss functions.
    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    """

    is_multi_class = False

    def __init__(self, n_classes):
        self.K = n_classes

    def init_estimator(self, X, y):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, pred):
        """Compute the loss of prediction ``pred`` and ``y``. """

    @abstractmethod
    def negative_gradient(self, y, y_pred, **kargs):
        """Compute the negative gradient.
        Parameters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_mask, learning_rate=1.0, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.
        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : np.ndarray, shape=(n, m)
            The data array.
        y : np.ndarray, shape=(n,)
            The target labels.
        residual : np.ndarray, shape=(n,)
            The residuals (usually the negative gradient).
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         y_pred[:, k])

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += (learning_rate
                         * tree.value[:, 0, 0].take(terminal_regions, axis=0))

    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        """Template method for updating terminal regions (=leaves). """

class RegressionLossFunction(six.with_metaclass(ABCMeta, LossFunction)):
    """Base class for regression loss functions. """

    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super(RegressionLossFunction, self).__init__(n_classes)
        

class NormalizedDiscountedCumulativeGain(RegressionLossFunction):
    """Quantify ranking by weighing more top higher ranked samples.
    Contrary to other subclasses of RegressionLossFunction, this is not a loss
    function to minimize but a score function to maximize.
    """
    def __init__(self, n_classes, max_rank=10):
        super(NormalizedDiscountedCumulativeGain, self).__init__(n_classes)
        self.max_rank = max_rank
        self.weights = None

    def init_estimator(self):
        return ZeroEstimator(is_classification=False)

    def _groupby(self, sample_group):
        fl = sample_group.flat
        start = 0
        start_group = fl.next()
        for g in fl:
            if start_group != g:
                yield start, fl.index - 1
                start = fl.index - 1
                start_group = g
        yield start, fl.index

    def __call__(self, y, pred, mask=Ellipsis, sample_group=None):
        pred = pred[mask].ravel()
        y = y[mask]
        if sample_group is None:
            ix = np.lexsort((y, -pred))
            ndcg = _ndcg(y[ix][:self.max_rank],
                         np.sort(y)[::-1][:self.max_rank])
        else:
            sample_group = sample_group[mask]
            n_group = 0
            s_ndcg = 0
            # for each group compute the ndcg
            for start, end in self._groupby(sample_group):
                ix = np.lexsort((y[start:end], -pred[start:end]))
                tmp_ndcg = _ndcg(y[ix + start][:self.max_rank],
                                 np.sort(y[start:end])[::-1][:self.max_rank])
                s_ndcg += tmp_ndcg
                n_group += 1

            ndcg = s_ndcg / n_group
        return ndcg

    def negative_gradient(self, y_true, y_pred, sample_group=None, **kargs):
        y_pred = y_pred.ravel()
        # the lambda terms
        grad = np.empty_like(y_true, dtype=np.float64)

        # for updating terminal regions
        self.weights = np.empty_like(y_true, dtype=np.float64)

        if sample_group is None:
            ix = np.lexsort((y_true, -y_pred))
            inv_ix = np.empty_like(ix)
            inv_ix[ix] = np.arange(len(ix))
            tmp_grad, tmp_weights = _lambda(y_true[ix], y_pred[ix],
                                            self.max_rank)
            grad = tmp_grad[inv_ix]
            self.weights = tmp_weights[inv_ix]
        else:
            for start, end in self._groupby(sample_group):
                ix = np.lexsort((y_true[start:end], -y_pred[start:end]))
                inv_ix = np.empty_like(ix)
                inv_ix[ix] = np.arange(len(ix))

                # sort by current score before passing
                # and then remap the return values
                tmp_grad, tmp_weights = _lambda(y_true[ix + start],
                                                y_pred[ix + start],
                                                self.max_rank)
                grad[start:end] = tmp_grad[inv_ix]
                self.weights[start:end] = tmp_weights[inv_ix]

        return grad

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        terminal_region = np.where(terminal_regions == leaf)[0]
        num = np.sum(residual.take(terminal_region, axis=0))
        den = np.sum(self.weights.take(terminal_region, axis=0))
        # Numerator is gradient and denominator is 2nd derivative
        # Need to be careful when denominator is zero.
        # If num == 0 and den == 0: set to zero
        # If num != 0 and den == 0: set den to epsilon
        tree.value[leaf, 0, 0] = num / (den + np.finfo(float).eps)


LOSS_FUNCTIONS ['ndcg']= NormalizedDiscountedCumulativeGain


class BaseRankBoosting(BaseGradientBoosting):
    """Abstract base class for Gradient Boosting. """

    @abstractmethod
    def __init__(self, loss, learning_rate, n_estimators, min_samples_split,
                 min_samples_leaf, max_depth, init, subsample, max_features,
                 random_state, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 max_rank=10, warm_start=False):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.max_rank = max_rank
        self.warm_start = warm_start

        self.estimators_ = np.empty((0, 0), dtype=np.object)
    def fit(self, X, y, monitor=None, **kargs):
        """Fit the gradient boosting model.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes
            ``0, 1, ..., n_classes_-1``
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshotting.
        sample_group : array-like, shape = [n_samples], optional (default=None)
            Indicates which group each sample belongs to. Samples from the
            same group must be adjoined. For example, [0, 0, 2, 2, 1, 1] is valid
            but [0, 1, 0] is invalid. If None, then all the samples will
            be considered to be of the same group.
        Returns
        -------
        self : object
            Returns self.
        """
        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        X, y = check_X_y(X, y, dtype=DTYPE)
        n_samples, n_features = X.shape
        self.n_features = n_features
        random_state = check_random_state(self.random_state)
        self._check_params()

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model
            self.init_.fit(X, y)

            # init predictions
            y_pred = self.init_.predict(X)
            begin_at_stage = 0
        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            y_pred = self.decision_function(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(X, y, y_pred, random_state,
                                    begin_at_stage, monitor, **kargs)
        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
            if hasattr(self, '_oob_score_'):
                self._oob_score_ = self._oob_score_[:n_stages]

        return self

    def _fit_stage(self, i, X, y, y_pred, sample_mask,
                   criterion, splitter, random_state, **kargs):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """
        loss = self.loss_
        original_y = y

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, y_pred, k=k, **kargs)

            # induce regression tree on residuals
            tree = DecisionTreeRegressor(
                criterion=criterion,
                splitter=splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state)

            sample_weight = None
            if self.subsample < 1.0:
                sample_weight = sample_mask.astype(np.float64)

            tree.fit(X, residual,
                     sample_weight=sample_weight, check_input=False)

            # update tree leaves
            loss.update_terminal_regions(tree.tree_, X, y, residual, y_pred,
                                         sample_mask, self.learning_rate, k=k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return y_pred

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss in ('mdeviance', 'bdeviance'):
            warn(("Loss '{0:s}' is deprecated as of version 0.14. "
                 "Use 'deviance' instead. ").format(self.loss))

        if self.loss == 'deviance':
            loss_class = (MultinomialDeviance
                          if len(self.classes_) > 2
                          else BinomialDeviance)
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

        if self.loss in ('huber', 'quantile'):
            self.loss_ = loss_class(self.n_classes_, self.alpha)
        elif self.loss in ('ndcg',):
            self.loss_ = loss_class(self.n_classes_, self.max_rank)
        else:
            self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            if isinstance(self.init, six.string_types):
                if self.init not in INIT_ESTIMATORS:
                    raise ValueError('init="%s" is not supported' % self.init)
            else:
                if (not hasattr(self.init, 'fit')
                        or not hasattr(self.init, 'predict')):
                    raise ValueError("init=%r must be valid BaseEstimator "
                                     "and support both fit and "
                                     "predict" % self.init)

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features)))
                else:
                    # is regression
                    max_features = self.n_features
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features)

        self.max_features_ = max_features

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """

        if self.init is None:
            self.init_ = self.loss_.init_estimator()
        elif isinstance(self.init, six.string_types):
            self.init_ = INIT_ESTIMATORS[self.init]()
        else:
            self.init_ = self.init

        self.estimators_ = np.empty((self.n_estimators, self.loss_.K),
                                    dtype=np.object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self._oob_score_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_improvement_ = np.zeros((self.n_estimators),
                                             dtype=np.float64)

    def _clear_state(self):
        """Clear the state of the gradient boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, '_oob_score_'):
            del self._oob_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes. """
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_.resize((total_n_estimators, self.loss_.K))
        self.train_score_.resize(total_n_estimators)
        if (self.subsample < 1
                or hasattr(self, '_oob_score_')
                or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, '_oob_score_'):
                self._oob_score_.resize(total_n_estimators)
            else:
                self._oob_score_ = np.zeros((total_n_estimators),
                                            dtype=np.float64)
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_.resize(total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0



    def _random_sample_mask(self, n_samples, n_inbag, random_state, **kargs):
        return _random_sample_mask(n_samples, n_inbag, random_state)

    def _inbag_samples(self, n_samples, **kargs):
        return max(1, int(self.subsample * n_samples))

    def _fit_stages(self, X, y, y_pred, random_state, begin_at_stage=0,
                    monitor=None, **kargs):
        """Iteratively fits the stages.
        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=np.bool)
        n_inbag = self._inbag_samples(n_samples, **kargs)
        loss_ = self.loss_


        # init criterion and splitter
        criterion = FriedmanMSE(1)
        splitter = PresortBestSplitter(criterion,
                                       self.max_features_,
                                       self.min_samples_leaf,
                                       0,#min_weight_leaf
                                       random_state)
        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        # perform boosting iterations
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = self._random_sample_mask(n_samples, n_inbag,
                                                       random_state, **kargs)
                # OOB score before adding this stage
                old_oob_score = loss_(y, y_pred, mask=~sample_mask, **kargs)

            # fit next stage of trees
            y_pred = self._fit_stage(i, X, y, y_pred, sample_mask,
                                     criterion, splitter, random_state, **kargs)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y, y_pred, mask=sample_mask,
                                             **kargs)
                self._oob_score_[i] = loss_(y, y_pred, mask=~sample_mask,
                                            **kargs)
                self.oob_improvement_[i] = old_oob_score - self._oob_score_[i]
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = self.loss_(y, y_pred, **kargs)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break
        return i + 1

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()




#mostly taken from jwkvam's https://github.com/jwkvam/scikit-learn/blob/lambdamart/sklearn/ensemble/gradient_boosting.py
class LambdaMART(BaseRankBoosting):

    _SUPPORTED_LOSS = ('ndcg',)

    def __init__(self, learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                 max_depth=3, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 max_rank=10, gain='exponential', warm_start=False):

        self.gain = gain

        super(LambdaMART, self).__init__(
            'ndcg', learning_rate, n_estimators, min_samples_split,
            min_samples_leaf, max_depth, init, subsample, max_features,
            random_state, alpha, verbose, max_leaf_nodes=max_leaf_nodes,
            max_rank=max_rank, warm_start=warm_start)

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        super(LambdaMART, self)._check_params()
        if self.max_rank is not None and self.max_rank <= 0:
            raise ValueError("max_rank must be None or a positive integer but "
                             "was %r" % self.max_rank)

    def _gain(self, y):
        if self.gain == 'exponential':
            y = 2 ** y - 1
        return y

    def _random_sample_mask(self, n_samples, n_inbag, random_state,
                            sample_group=None, **kargs):
        if sample_group is None:
            mask = super(LambdaMART, self)._random_sample_mask(n_samples,
                                                               n_inbag,
                                                               random_state)
        else:
            mask = _ranked_random_sample_mask(n_samples, n_inbag, random_state,
                                              sample_group, self.n_uniq_group)
        return mask

    def _inbag_samples(self, n_samples, sample_group=None, **kargs):
        if sample_group is None:
            n_inbag = max(1, int(self.subsample * n_samples))
        else:
            n_inbag = max(1, int(self.subsample * self.n_uniq_group))
        return n_inbag

    def fit(self, X, y,Q=None, monitor=None):

        if Q is None:
            Q = np.ones(Q)
        else:
            ids = np.argsort(Q)
            X,y,Q = X[ids],y[ids],Q[ids]
        self.n_classes_ = 1
        if Q is not None:
            Q = column_or_1d(Q, warn=True)
            # check if sample_group is grouped
            uniq_group = {Q[0]}
            last_group = Q[0]
            for g in Q[1:]:
                if g != last_group:
                    # group must be unseen thus far
                    if g in uniq_group:
                        raise ValueError("queries must be grouped together")
                    uniq_group.add(g)
                    last_group = g
            self.n_uniq_group = len(uniq_group)
        y = self._gain(column_or_1d(y, warn=True))
        return super(LambdaMART, self).fit(X, y, monitor,sample_group=Q)

    def predict(self, X):
        return self.decision_function(X).ravel()

    def staged_predict(self, X):
        for y in self.staged_decision_function(X):
            yield y.ravel()

    def score(self, X, y, sample_group=None):
        y = self._gain(column_or_1d(y, warn=True))
        return self.loss_(y, self.decision_function(X), sample_group)
        
        
        
