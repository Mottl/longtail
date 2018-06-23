# -*- coding: utf-8 -*-
"""
Transforms RV from the given empirical distribution to the standard normal distribution
Author: Dmitry Mottl (https://github.com/Mottl)
License: MIT
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def fit_distributions(y, distributions=None, verbose=False):
    """
    fit distributions to data `y`

    Parameters
    ----------
    y : array

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
        params[name] = distr.fit(y)
        if verbose:
            print(name, params[name])
    return params


def plot(y, y_name=None, params=None, **kwargs):
    """
    plot probability distribution function for `y`
    and overlay distributions calculated with `params`

    Parameters
    ----------
    y : array

    params: list of best-fit parameters returned by fit_distributions() function

    Return value
    ------------
    params from fit_distributions() function
    """

    if y is not np.ndarray:
        y = np.array(y)
    if params is None:
        print("Estimating distributions parameters...")
        params = fit_distributions(y, verbose=True)

    label = y_name or "data"

    # plot PDF
    y_min = np.percentile(y, 0.9)
    y_max = -np.percentile(-y, 0.9)
    y_ = y[(y>=y_min) & (y<=y_max)]
    num_bins = int(np.log(len(y_))*5)
    y_space = np.linspace(y_min, y_max, 1000)

    f, ax = plt.subplots(**kwargs)
    ax.hist(y_, bins=num_bins, density=True, alpha=0.33, color="dodgerblue", label=label)
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(y_space, distr.pdf(y_space, loc=param[0], scale=param[1]), label=name)

    ax.legend()
    ax.set_ylabel('pdf')
    if y_name is not None:
        ax.set_xlabel(y_name)
    ax.grid(True)
    plt.show()

    # plot LOG PDF
    y_min, y_max = y.min(), y.max()

    num_bins = int(np.log(len(y))*5)
    y_space = np.linspace(y_min, y_max, 1000)

    bins_means = []  # mean of bin interval
    bins_ys = []  # number of ys in interval

    y_step = (y_max - y_min) / num_bins
    for y_left in np.arange(y_min, y_max, y_step):
        bins_means.append(y_left + y_step/2.)
        bins_ys.append(np.sum((y>=y_left) & (y<y_left+y_step)))
    bins_ys = np.array(bins_ys) / len(y) / y_step  # normalize

    f, ax = plt.subplots(**kwargs)
    ax.scatter(bins_means, bins_ys, s=5., color="dodgerblue", label=label)
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(y_space, distr.pdf(y_space, loc=param[0], scale=param[1]), label=name)

    ax.legend()
    ax.set_ylabel('pdf')
    ax.set_yscale('log')
    if y_name is not None:
        ax.set_xlabel(y_name)
    ax.grid(True)
    plt.show()

    return params


class GaussianScaler():
    """Transform data to make it Gaussian distributed."""

    def __init__(self):
        self.transform_table = None

    def fit(self, X, y=None, bins='auto'):
        """Compute empirical parameters for transforming the data to Gaussian distribution."""

        if len(X.shape)>1:
            raise NotImplementedError("X must be an 1d-array")

        X_sorted = np.sort(X)
        x_min, x_max = X_sorted[0], X_sorted[-1]
        x_total = len(X)

        if bins == 'auto':
            bins = int(np.log(x_total)*5)

        x_step = (x_max - x_min) / bins

        self.transform_table = []
        self.transform_table.append((-np.inf, -np.inf, 0.))
        for x in np.arange(x_min + x_step, x_max + x_step, x_step):
            cdf_empiric = np.sum(X_sorted < x) / x_total
            if cdf_empiric == 1:
                break

            # use probit function to get correspoding x from standard norm. distribution:
            x_norm = stats.norm.ppf(cdf_empiric)
            self.transform_table.append((x, x_norm, 0.))
        self.transform_table.append((np.inf, np.inf, 0.))

        self.transform_table = np.array(self.transform_table)

        # compute x -> x_norm coefficients
        dx = self.transform_table[2:-1, 0] - self.transform_table[1:-2, 0]
        dx_norm = self.transform_table[2:-1, 1] - self.transform_table[1:-2, 1]
        self.transform_table[2:-1, 2] = dx_norm / dx

        """
        # generic non-optimized code would look like this:
        for i in range(2, len(self.transform_table) - 1):
            dx = self.transform_table[i, 0] - self.transform_table[i-1, 0]
            dx_norm = self.transform_table[i, 1] - self.transform_table[i-1, 1]
            self.transform_table[i, 2] = dx_norm / dx
        """

        # fill boundary bins (plus/minus infinity) intervals:
        self.transform_table[0,  2] = self.transform_table[2,  2]
        self.transform_table[1,  2] = self.transform_table[2,  2]
        self.transform_table[-1, 2] = self.transform_table[-2, 2]

    def transform(self, X, y=None):
        """Transform X to Gaussian distributed (standard normal)."""

        if self.transform_table is None:
            raise Exception(("This GaussianScaler instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this method."))

        def _transform(x):
            # x(empirical) -> x(normaly distributed)
            lefts  = self.transform_table[self.transform_table[:, 0] <  x]
            rights = self.transform_table[self.transform_table[:, 0] >= x]

            left_boundary = lefts[-1]
            right_boundary = rights[0]

            k = right_boundary[2]

            if right_boundary[0] == np.inf:
                x_norm = left_boundary[1] + k * (x - left_boundary[0])
            else:
                x_norm = right_boundary[1] + k * (x - right_boundary[0])

            return x_norm

        vtransform = np.vectorize(_transform)
        return vtransform(X)

    def fit_transform(self, X, y=None):
        """ Fit to data, then transform it."""

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Transform back the data from Gaussian to the original empirical distribution."""

        if self.transform_table is None:
            raise Exception(("This GaussianScaler instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this method."))

        def _inverse_transform(x):
            # x(normaly distributed) -> x(empirical)
            lefts  = self.transform_table[self.transform_table[:, 1] <  x]
            rights = self.transform_table[self.transform_table[:, 1] >= x]

            left_boundary = lefts[-1]
            right_boundary = rights[0]

            k = right_boundary[2]

            if right_boundary[1] == np.inf:
                x_emp = left_boundary[0] + (x - left_boundary[1]) / k
            else:
                x_emp = right_boundary[0] + (x - right_boundary[1]) / k

            return x_emp

        vinverse_transform = np.vectorize(_inverse_transform)
        return vinverse_transform(X)
