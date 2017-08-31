"""
The :mod:`tslearn.barycenters` module gathers algorithms for time series barycenter computation.
"""

# Code for soft DTW is by Mathieu Blondel under Simplified BSD license

import numpy
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from tslearn.utils import to_time_series_dataset, check_equal_size
from tslearn.metrics import dtw_path, SquaredEuclidean, SoftDTW
from tslearn.preprocessing import TimeSeriesResampler


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _set_weights(w, X):
    if w is None or len(w) != len(X):
        w = numpy.ones((X.shape[0],))
    return w


class EuclideanBarycenter:
    """Standard Euclidean barycenter computed from a set of time series.

    Parameters
    ----------
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> bar = EuclideanBarycenter().fit(time_series)
    >>> bar.shape
    (4, 1)
    >>> bar  # doctest: +ELLIPSIS
    array([[ 1. ],
           [ 2. ],
           [ 3.5],
           [ 4.5]])

    Note
    ----
    This algorithm requires equal sized time series.
    """
    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X):
        """Compute the barycenter from a dataset of time series.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        numpy.array of shape (sz, d)
            Barycenter of the provided time series dataset.
        """
        X_ = to_time_series_dataset(X)
        self.weights = _set_weights(self.weights, X_)
        return numpy.average(X_, axis=0, weights=self.weights)


class DTWBarycenterAveraging(EuclideanBarycenter):
    """DTW Barycenter Averaging (DBA) method.

    DBA was originally presented in [1]_.

    Parameters
    ----------
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
    max_iter : int (default: 30)
        Number of iterations of the EM procedure.
    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter is that of the data provided at fit
        time or that of the initial barycenter if specified.
    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the EM process.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower than this value, the EM procedure stops.
    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> euc_bar = EuclideanBarycenter().fit(time_series)
    >>> dba_bar = DTWBarycenterAveraging(max_iter=0).fit(time_series)
    >>> dba_bar.shape
    (4, 1)
    >>> numpy.alltrue(numpy.abs(euc_bar - dba_bar) < 1e-9)
    True
    >>> DTWBarycenterAveraging(max_iter=0, barycenter_size=5).fit(time_series).shape
    (5, 1)
    >>> DTWBarycenterAveraging(max_iter=5, barycenter_size=5, verbose=True).fit(time_series).shape  # doctest: +ELLIPSIS
    [DBA] epoch 1, cost: ...
    (5, 1)
    
    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method for dynamic time warping, with
       applications to clustering. Pattern Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    def __init__(self, weights=None, max_iter=30, barycenter_size=None, init_barycenter=None, tol=1e-5, verbose=False):
        EuclideanBarycenter.__init__(self, weights=weights)
        self.max_iter = max_iter
        self.init_barycenter = init_barycenter
        if init_barycenter is not None:
            self.barycenter_size = init_barycenter.shape[0]
        else:
            self.barycenter_size = barycenter_size
        self.tol = tol
        self.verbose = verbose

    def fit(self, X):
        """Compute the barycenter from a dataset of time series.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size is None
            DBA barycenter of the provided time series dataset.
        """
        X_ = to_time_series_dataset(X, equal_size=False)
        if self.barycenter_size is None:
            self.barycenter_size = X_[0].shape[0]
        self.weights = _set_weights(self.weights, X_)
        if self.init_barycenter is None:
            barycenter = self._init_avg(X_)
        else:
            barycenter = self.init_barycenter
        cost_prev, cost = numpy.inf, numpy.inf
        for it in range(self.max_iter):
            assign = self._petitjean_assignment(X_, barycenter)
            cost = self._petitjean_cost(X_, barycenter, assign)
            if self.verbose:
                print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
            barycenter = self._petitjean_update_barycenter(X_, assign)
            if cost_prev < cost:
                raise ValueError
            if cost_prev - cost < self.tol:
                break
            else:
                cost_prev = cost
        return barycenter

    def _init_avg(self, X):
        if X[0].shape[0] == self.barycenter_size and check_equal_size(X):
            return X.mean(axis=0)
        else:
            X_ = TimeSeriesResampler(sz=self.barycenter_size).fit_transform(X)
            return X_.mean(axis=0)

    def _petitjean_assignment(self, X, barycenter):
        n = X.shape[0]
        assign = ([[] for _ in range(self.barycenter_size)], [[] for _ in range(self.barycenter_size)])
        for i in range(n):
            path, _ = dtw_path(X[i], barycenter)
            for pair in path:
                assign[0][pair[1]].append(i)
                assign[1][pair[1]].append(pair[0])
        return assign

    def _petitjean_update_barycenter(self, X, assign):
        barycenter = numpy.zeros((self.barycenter_size, X.shape[-1]))
        for t in range(self.barycenter_size):
            barycenter[t] = numpy.average(X[assign[0][t], assign[1][t]], axis=0, weights=self.weights[assign[0][t]])
        return barycenter

    def _petitjean_cost(self, X, barycenter, assign):
        cost = 0.
        for t_barycenter in range(self.barycenter_size):
            for i_ts, t_ts in zip(assign[0][t_barycenter], assign[1][t_barycenter]):
                cost += numpy.linalg.norm(X[i_ts, t_ts] - barycenter[t_barycenter]) ** 2
        return cost / X.shape[0]


class SoftDTWBarycenter(EuclideanBarycenter):
    """Compute barycenter (time series averaging) under the soft-DTW geometry.

    Parameters
    ----------
    gamma: float
        Regularization parameter.
        Lower is less smoothed (closer to true DTW).
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
    method: string
        Optimization method, passed to `scipy.optimize.minimize`.
        Default: L-BFGS.
    tol: float
        Tolerance of the method used.
    max_iter: int
        Maximum number of iterations.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> euc_bar = EuclideanBarycenter().fit(time_series)
    >>> stdw_bar = SoftDTWBarycenter(max_iter=0).fit(time_series)
    >>> stdw_bar.shape
    (4, 1)
    >>> numpy.alltrue(numpy.abs(euc_bar - stdw_bar) < 1e-9)
    True
    >>> SoftDTWBarycenter(max_iter=5).fit(time_series).shape
    (4, 1)
    """
    def __init__(self, gamma=1.0, weights=None, method="L-BFGS-B", tol=1e-3, max_iter=50, init=None):
        EuclideanBarycenter.__init__(self, weights=weights)
        self.method = method
        self.tol = tol
        self.gamma = gamma
        self.max_iter = max_iter
        self._X_fit = None
        if init is None:
            self.barycenter_ = None
        else:
            self.barycenter_ = init

    def _func(self, Z):
        # Compute objective value and grad at Z.

        Z = Z.reshape(self.barycenter_.shape)

        G = numpy.zeros_like(Z)

        obj = 0

        for i in range(len(self._X_fit)):
            D = SquaredEuclidean(Z, self._X_fit[i])
            sdtw = SoftDTW(D, gamma=self.gamma)
            value = sdtw.compute()
            E = sdtw.grad()
            G_tmp = D.jacobian_product(E)
            G += self.weights[i] * G_tmp
            obj += self.weights[i] * value

        return obj, G.ravel()

    def fit(self, X):
        self._X_fit = to_time_series_dataset(X, equal_size=False)
        self.weights = _set_weights(self.weights, self._X_fit)
        if self.barycenter_ is None:
            self.barycenter_ = EuclideanBarycenter.fit(self, self._X_fit)

        if self.max_iter > 0:
            # The function works with vectors so we need to vectorize barycenter_.
            res = minimize(self._func, self.barycenter_.ravel(), method=self.method, jac=True, tol=self.tol,
                           options=dict(maxiter=self.max_iter, disp=False))
            return res.x.reshape(self.barycenter_.shape)
        else:
            return self.barycenter_
