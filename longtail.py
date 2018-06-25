# -*- coding: utf-8 -*-
"""
Transforms RV from the given empirical distribution to the standard normal distribution
Author: Dmitry Mottl (https://github.com/Mottl)
License: MIT
"""

import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

try:
    import pandas as pd
except:
    pass

def fit_distributions(X, distributions=None, verbose=False):
    """
    fit distributions to data `X`

    Parameters
    ----------
    X : 1d-array

    distributions : array of strings (['norm', 'laplace', etc..])
    defaults distributions are: ['norm', 'laplace', 'cauchy']

    verbose: bool, default False
    """

    if distributions is None:
        distributions = [
            'norm', 'laplace', 'cauchy'
        ]

    params = {}
    for name in distributions:
        distr = getattr(stats, name)
        params[name] = distr.fit(X)
        if verbose:
            print(name, params[name])
    return params


def plot(X, X_name=None, params=None, **kwargs):
    """
    plot probability distribution function for `y`
    and overlay distributions calculated with `params`

    Parameters
    ----------
    x : array

    params: list of best-fit parameters returned by fit_distributions() function

    Return value
    ------------
    params from fit_distributions() function
    """

    if X is not np.ndarray:
        X = np.array(X)
    if params is None:
        print("Estimating distributions parameters...")
        params = fit_distributions(X, verbose=True)

    label = X_name or "data"

    # plot PDF
    x_min = np.percentile(X, 0.9)
    x_max = -np.percentile(-X, 0.9)
    X_ = X[(X>=x_min) & (X<=x_max)]
    num_bins = int(np.log(len(X_))*5)
    x_space = np.linspace(x_min, x_max, 1000)

    f, ax = plt.subplots(**kwargs)
    ax.hist(X_, bins=num_bins, density=True, alpha=0.33, color="dodgerblue", label=label)
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(x_space, distr.pdf(x_space, loc=param[0], scale=param[1]), label=name)

    ax.legend()
    ax.set_ylabel('pdf')
    if X_name is not None:
        ax.set_xlabel(X_name)
    ax.grid(True)
    plt.show()

    # plot LOG PDF
    x_min, x_max = X.min(), X.max()

    num_bins = int(np.log(len(X))*5)
    x_space = np.linspace(x_min, x_max, 1000)

    bins_means = []  # mean of bin interval
    bins_xs = []  # number of ys in interval

    x_step = (x_max - x_min) / num_bins
    for x_left in np.arange(x_min, x_max, x_step):
        bins_means.append(x_left + x_step/2.)
        bins_xs.append(np.sum((X>=x_left) & (X<x_left+x_step)))
    bins_xs = np.array(bins_xs) / len(X) / x_step  # normalize

    f, ax = plt.subplots(**kwargs)
    ax.scatter(bins_means, bins_xs, s=5., color="dodgerblue", label=label)
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(x_space, distr.pdf(x_space, loc=param[0], scale=param[1]), label=name)

    ax.legend()
    ax.set_ylabel('pdf')
    ax.set_yscale('log')
    if X_name is not None:
        ax.set_xlabel(X_name)
    ax.grid(True)
    plt.show()

    return params


class GaussianScaler():
    """Transform data to make it Gaussian distributed."""

    def __init__(self):
        self.transform_table = None
        self.features_names = None
        self.__num_vars = None

    def fit(self, X, y=None):
        """Compute empirical parameters for transforming the data to Gaussian distribution.

        Parameters:
        -----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to fit
        """

        if len(X.shape)>2:
            raise NotImplementedError("X must be an 1d-array or a 2d-matrix of observations x features")

        # convert from pd.DataFrame to np.ndarrray:
        if "pandas.core.frame" in sys.modules.keys() and type(X) == pd.core.frame.DataFrame:
            self.features_names = X.columns.values
            X = X.values

        if len(X.shape) == 2 and X.shape[1] == 1:
            X = X.ravel()

        if X.dtype != float:
            raise Exception("X.dtype is {}, but should be float".format(X.dtype))

        self.__num_vars = len(X.shape)
        self.transform_table = []

        for j in range(self.__num_vars):
            if self.__num_vars > 1:
                X_sorted = np.array(np.unique(X[:, j], return_counts=True)).T
            else:
                X_sorted = np.array(np.unique(X, return_counts=True)).T
            X_sorted[:, 1] = np.cumsum(X_sorted[:, 1])
            total = X_sorted[-1,1]
            # X_sorted[:, 0] is x, X_sorted[:, 1] is the number of occurences <= x

            STEP_MULT = 0.1  # step multiplier
            MIN_STEP = 5

            step = MIN_STEP
            i = step
            prev_x = -np.inf

            transform_table = []
            transform_table.append((-np.inf, -np.inf, 0.))

            while True:
                index = np.argmax(X_sorted[:,1] >= i)
                row = X_sorted[index]
                x = row[0]
                if x != prev_x:
                    cdf_empiric = row[1] / total
                    x_norm = stats.norm.ppf(cdf_empiric)

                    if x_norm == np.inf:  # too large - stop
                        break
                    if x_norm != -np.inf:
                        transform_table.append((x, x_norm, 0.))

                    if cdf_empiric < 0.5:
                        step = int(row[1] * STEP_MULT)
                    else:
                        step = int((total - row[1]) * STEP_MULT)

                    step = max(step, MIN_STEP)
                    prev_x = x

                i = i + step
                if i >= total:
                    break

            transform_table.append((np.inf, np.inf, 0.))
            transform_table = np.array(transform_table)

            # compute x -> x_norm coefficients
            dx = transform_table[2:-1, 0] - transform_table[1:-2, 0]
            dx_norm = transform_table[2:-1, 1] - transform_table[1:-2, 1]
            transform_table[2:-1, 2] = dx_norm / dx

            """
            # generic non-optimized code would look like this:
            for i in range(2, len(transform_table) - 1):
                dx = transform_table[i, 0] - transform_table[i-1, 0]
                dx_norm = transform_table[i, 1] - transform_table[i-1, 1]
                transform_table[i, 2] = dx_norm / dx
            """

            # fill boundary bins (plus/minus infinity) intervals:
            transform_table[0,  2] = transform_table[2,  2]
            transform_table[1,  2] = transform_table[2,  2]
            transform_table[-1, 2] = transform_table[-2, 2]

            # add current transform table for the feature to self.transform_table
            self.transform_table.append(transform_table)

    def transform(self, X, y=None):
        """Transform X to Gaussian distributed (standard normal).

        Parameters
        ----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to transform
        """

        if self.transform_table is None:
            raise Exception(("This GaussianScaler instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this method."))

        if len(X.shape)>2:
            raise NotImplementedError("X must be an 1d-array or a 2d-matrix of observations x features")

        # convert from pd.DataFrame to np.ndarrray:
        if "pandas.core.frame" in sys.modules.keys() and type(X) == pd.core.frame.DataFrame:
            features_names = X.columns.values
            if (features_names != self.features_names).any():
                raise Exception("Feature names mismatch.\nFeatures for fit():{}\nFeatures for transform:{}".format(
                    self.features_names, features_names))
            save_index = X.index.copy()
            X = X.values.copy()

        if len(X.shape) == 2 and X.shape[1] == 1:
            X = X.ravel()

        if X.dtype != float:
            raise Exception("X.dtype is {}, but should be float".format(X.dtype))

        num_vars = len(X.shape)
        if self.__num_vars != num_vars:
            raise Exception("Number of features mismatch for fit() and transform(): {} vs {}".format(
                self.__num_vars, num_vars))

        def _transform(x, j):
            # x(empirical) -> x(normaly distributed)
            transform_table = self.transform_table[j]
            lefts  = transform_table[transform_table[:, 0] <  x]
            rights = transform_table[transform_table[:, 0] >= x]

            left_boundary = lefts[-1]
            right_boundary = rights[0]

            k = right_boundary[2]

            if right_boundary[0] == np.inf:
                x_norm = left_boundary[1] + k * (x - left_boundary[0])
            else:
                x_norm = right_boundary[1] + k * (x - right_boundary[0])

            return x_norm

        vtransform = np.vectorize(_transform)

        # transform all features:
        for j in range(self.__num_vars):
            X[:, j] = vtransform(X[:, j], j)

        # reconstruct X as a DataFrame:
        if self.features_names is not None:
            X = pd.DataFrame(X, columns=self.features_names, index=save_index)

        return X

    def fit_transform(self, X, y=None):
        """ Fit to data, then transform it.

        Parameters
        ----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to transform
        """

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Transform back the data from Gaussian to the original empirical distribution.

        Parameters
        ----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to inverse transform
        """

        if self.transform_table is None:
            raise Exception(("This GaussianScaler instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this method."))

        if len(X.shape)>2:
            raise NotImplementedError("X must be an 1d-array or a 2d-matrix of observations x features")

        # convert from pd.DataFrame to np.ndarrray:
        if "pandas.core.frame" in sys.modules.keys() and type(X) == pd.core.frame.DataFrame:
            features_names = X.columns.values
            if (features_names != self.features_names).any():
                raise Exception("Feature names mismatch.\nFeatures for fit():{}\nFeatures for transform:{}".format(
                    self.features_names, features_names))
            save_index = X.index.copy()
            X = X.values.copy()

        if len(X.shape) == 2 and X.shape[1] == 1:
            X = X.ravel()

        if X.dtype != float:
            raise Exception("X.dtype is {}, but should be float".format(X.dtype))

        num_vars = len(X.shape)
        if self.__num_vars != num_vars:
            raise Exception("Number of features mismatch for fit() and transform(): {} vs {}".format(
                self.__num_vars, num_vars))

        def _inverse_transform(x, j):
            # x(normaly distributed) -> x(empirical)
            transform_table = self.transform_table[j]
            lefts  = transform_table[transform_table[:, 1] <  x]
            rights = transform_table[transform_table[:, 1] >= x]

            left_boundary = lefts[-1]
            right_boundary = rights[0]

            k = right_boundary[2]

            if right_boundary[1] == np.inf:
                x_emp = left_boundary[0] + (x - left_boundary[1]) / k
            else:
                x_emp = right_boundary[0] + (x - right_boundary[1]) / k

            return x_emp

        vinverse_transform = np.vectorize(_inverse_transform)

        # transform all features:
        for j in range(self.__num_vars):
            X[:, j] = vinverse_transform(X[:, j], j)

        # reconstruct X as a DataFrame:
        if self.features_names is not None:
            X = pd.DataFrame(X, columns=self.features_names, index=save_index)

        return X
