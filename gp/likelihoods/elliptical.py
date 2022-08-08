# -*- coding: utf-8 -*-
import numpy as np


def elliptical(f, log_likelihood, L):
    """elipitical sampling

    f is Gaussian Process
    f ~ N(0, K)

    Args:
        f : target distribution
        log_likelihood : log likelihood of f
        L : triangle matrix of K

    """
    rho = log_likelihood(f) + np.log(np.random.uniform(0, 1))
    nu = np.dot(L, np.random.randn(len(f)))

    theta = np.random.uniform(0, 2 * np.pi)
    st, ed = theta - 2 * np.pi, theta

    while True:
        f = f * np.cos(theta) + nu * np.sin(theta)
        if log_likelihood(f) > rho:
            return f, log_likelihood(f)
        else:
            if theta > 0:
                ed = theta
            elif theta < 0:
                st = theta
            else:
                raise IOError('Slice sampling shrunk to the current position.')
            theta = np.random.uniform(st, ed)
