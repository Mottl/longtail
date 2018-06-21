# -*- coding: utf-8 -*-
"""
Checks if distribution is Gaussian
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
    """

    if y is not np.ndarray:
        y = np.array(y)
    if params is None:
        print("Estimating distributions parameters...")
        params = fit_distributions(y, verbose=True)

    # plot PDF
    y_min = np.percentile(y, 0.9)
    y_max = -np.percentile(-y, 0.9)
    y_ = y[(y>=y_min) & (y<=y_max)]
    num_bins = int(np.log(len(y_))*5)
    y_space = np.linspace(y_min, y_max, 1000)

    f, ax = plt.subplots(**kwargs)
    ax.hist(y_, bins=num_bins, density=True, alpha=0.5, color="dodgerblue")
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
    y_min = np.percentile(y, 0.1)
    y_max = -np.percentile(-y, 0.1)
    y_ = y[(y>=y_min) & (y<=y_max)]
    num_bins = int(np.log(len(y_))*5)
    y_space = np.linspace(y_min, y_max, 1000)

    bins_means = []  # mean of bin interval
    bins_ys = []  # number of ys in interval

    y_step = (y_max - y_min) / num_bins
    for y_left in np.arange(y_min, y_max, y_step):
        bins_means.append(y_left + y_step/2.)
        bins_ys.append(np.sum((y>=y_left) & (y<y_left+y_step)))
    bins_ys = np.log(np.array(bins_ys) / len(y) / y_step)  # normalize

    f, ax = plt.subplots(**kwargs)
    ax.scatter(bins_means, bins_ys, s=5., color="dodgerblue", label="data")
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(y_space, np.log(distr.pdf(y_space, loc=param[0], scale=param[1])), label=name)

    ax.legend()
    ax.set_ylabel('log pdf')
    if y_name is not None:
        ax.set_xlabel(y_name)
    ax.grid(True)
    plt.show()

    # plot LOG PDF
    y_min = np.percentile(y, 0.002)
    y_max = -np.percentile(-y, 0.002)
    y_ = y[(y>=y_min) & (y<=y_max)]
    num_bins = int(np.log(len(y_))*5)
    y_space = np.linspace(y_min, y_max, 1000)

    bins_means = []  # mean of bin interval
    bins_ys = []  # number of ys in interval

    y_step = (y_max - y_min) / num_bins
    for y_left in np.arange(y_min, y_max, y_step):
        bins_means.append(y_left + y_step/2.)
        bins_ys.append(np.sum((y>=y_left) & (y<y_left+y_step)))
    bins_ys = np.log(np.array(bins_ys) / len(y) / y_step)  # normalize

    f, ax = plt.subplots(**kwargs)
    ax.scatter(bins_means, bins_ys, s=5., color="dodgerblue", label="data")
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(y_space, np.log(distr.pdf(y_space, loc=param[0], scale=param[1])), label=name)

    ax.legend()
    ax.set_ylabel('log pdf')
    if y_name is not None:
        ax.set_xlabel(y_name)
    ax.grid(True)
    plt.show()
