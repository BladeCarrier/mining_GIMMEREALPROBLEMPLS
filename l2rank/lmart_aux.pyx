# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer
#
# Licence: BSD 3 clause

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport exp, log


ctypedef np.int32_t int32
ctypedef np.float64_t float64
ctypedef np.int8_t int8
ctypedef fused all32_64_t:
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

from numpy import bool as np_bool

# no namespace lookup for numpy dtype and array creation
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import bool as np_bool
from numpy import int8 as np_int8
from numpy import int32 as np_int32
from numpy import intp as np_intp
from numpy import float32 as np_float32
from numpy import float64 as np_float64

# Define a datatype for the data array
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.npy_intp SIZE_t


# constant to mark tree leafs
cdef int LEAF = -1

def _ranked_random_sample_mask(int n_total_samples, int n_total_in_bag,
                               random_state, cython.integral [::1] sample_group,
                               int n_uniq_group):
    """Create a random sample mask where ``n_total_in_bag`` elements are set.
    Parameters
    ----------
    n_total_samples : int
        The length of the resulting mask.
    n_total_in_bag : int
        The number of elements in the sample mask which are set to 1.
    sample_group : sample_group associated with each sample
    n_uniq_group : number of unique queries
    random_state : np.RandomState
        A numpy ``RandomState`` object.
    Returns
    -------
    sample_mask : np.ndarray, shape=[n_total_samples]
        An ndarray where ``n_total_in_bag`` elements are set to ``True``
        the others are ``False``.
    """
    cdef np.ndarray[float64, ndim=1, mode="c"] rand = \
         random_state.rand(n_uniq_group)
    cdef np.ndarray[int8, ndim=1, mode="c"] sample_mask = \
         np_zeros((n_total_samples,), dtype=np_int8)
    cdef np.ndarray[int32, ndim=1, mode="c"] group_mask = \
         np_zeros((n_total_in_bag,), dtype=np_int32)

    cdef int n_bagged = 0
    cdef int i = 0
    cdef int j = 0
    cdef int8 mask = 0
    cdef int32 last_group = 0

    last_group = sample_group[0]
    if rand[0] * n_uniq_group < n_total_in_bag - n_bagged:
        sample_mask[0] = 1
        mask = 1
        n_bagged += 1

    for i in range(1, n_total_samples):
        if sample_group[i] != last_group:
            last_group = sample_group[i]
            # track number of unique queries processed
            j += 1
            if rand[j] * (n_uniq_group - j) < (n_total_in_bag - n_bagged):
                mask = 1
                n_bagged += 1
            else:
                mask = 0
        sample_mask[i] = mask

    return sample_mask.astype(np_bool)


def _ndcg(all32_64_t [::1] y, all32_64_t [:] y_sorted):
    """Computes Normalized Discounted Cumulative Gain
    Currently there is no iteration cap.
    """
    cdef int i
    cdef double dcg = 0
    cdef double max_dcg = 0
    for i in range(y.shape[0]):
        dcg += y[i] / log(2 + i)
        max_dcg += y_sorted[i] / log(2 + i)
    if max_dcg == 0:
        return 1.
    return dcg / max_dcg


def _max_dcg(all32_64_t [:] y_sorted):
    """Computes Maximum Discounted Cumulative Gain
    """
    cdef int i
    cdef double max_dcg = 0
    for i in range(y_sorted.shape[0]):
        max_dcg += y_sorted[i] / log(2 + i)
    return max_dcg


def _lambda(all32_64_t [::1] y_true, double [::1] y_pred,
            max_rank):
    """Computes the Lambda-gradient and second derivatives as part of
    the LambdaMART algorithm.
    """
    cdef int i
    cdef int j

    cdef double [::1] grad = np_zeros(y_true.shape[0])
    cdef double [::1] weight = np_zeros(y_true.shape[0])
    cdef double score_diff
    cdef double ndcg_diff
    cdef double rho
    cdef double max_dcg
    cdef int sign

    if max_rank is None:
        max_rank = len(y_true)
    max_dcg = _max_dcg(np.sort(y_true)[::-1][:max_rank])
    cdef double ndcg = 0
    if max_dcg != 0:
        for i in range(max_rank):
            for j in range(i + 1, y_true.shape[0]):
                if y_true[i] != y_true[j]:
                    if j < max_rank:
                        ndcg_diff = ((y_true[j] - y_true[i]) / log(2 + i)
                                     + (y_true[i] - y_true[j]) / log(2 + j))
                    else:
                        ndcg_diff = (y_true[j] - y_true[i]) / log(2 + i)

                    ndcg_diff = abs(ndcg_diff / max_dcg)

                    score_diff = y_pred[i] - y_pred[j]
                    sign = 1 if y_true[i] > y_true[j] else -1
                    rho = 1 / (1 + exp(sign * score_diff))
                    grad[i] += sign * ndcg_diff * rho
                    grad[j] -= sign * ndcg_diff * rho
                    weight[i] += ndcg_diff * rho * (1 - rho)
                    weight[j] += ndcg_diff * rho * (1 - rho)

    return grad.base, weight.base