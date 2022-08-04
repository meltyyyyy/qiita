#!/usr/local/bin/python

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
from pylab import *
from numpy import exp, log
from numpy.linalg import cholesky as chol


def gpr_cauchy(f, param):
    x, y, gamma, Kinv = param
    M = len(x)
    return gpr_cauchy_lik(y[0:M], f[0:M], gamma, Kinv)


def gpr_cauchy_lik(y, f, gamma, Kinv):
    return - np.sum(log(gamma + (y - f)**2 / gamma)) \
           - np.dot(f, np.dot(Kinv, f)) / 2


def kgauss(tau, sigma):
    return lambda x, y: exp(tau) * exp(-(x - y)**2 / exp(sigma))


def kernel_matrix(xx, kernel):
    N = len(xx)
    eta = 1e-6
    return np.array(
        [kernel(xi, xj) for xi in xx for xj in xx]
    ).reshape(N, N) + eta * np.eye(N)


def gpr_mcmc(x, y, iters, xmin, xmax, gamma):
    xx = np.hstack((x, np.linspace(xmin, xmax, 100)))
    M = len(x)
    N = len(xx)
    K = kernel_matrix(xx, kgauss(1, 1))
    Kinv = inv(K[0:M, 0:M])
    S = chol(K)
    f = np.dot(S, randn(N))
    g = np.zeros(len(xx))
    for iter in xrange(iters):
        f, lik = elliptical(f, S, gpr_cauchy, (x, y, gamma, Kinv))
        g = g + f
        plot(xx[M:], f[M:])  # color='gray')
    plot(x, y, 'bx', markersize=14)
    plot(xx[M:], g[M:] / iters, 'k', linewidth=3)


def main():
    xmin = -5
    xmax = 5
    ymin = -7.5
    ymax = 12.5
    gamma = 0.2

    [x, y, f] = np.loadtxt('/Users/takeru.abe/Development/qiita/bo/gp/likelihoods/gpr-cauchy.txt').T
    iters = int(100)

    gpr_mcmc(x, y, iters, xmin, xmax, gamma)
    axis([xmin, xmax, ymin, ymax])


def elliptical(xx, prior, likfun, params=(), curlik=None, angle=0):
    # initialize
    D = len(xx)
    if curlik is None:
        curlik = likfun(xx, params)
    # set up the ellipse
    nu = np.dot(prior, randn(D))
    hh = log(rand()) + curlik
    # set up the bracket
    if angle <= 0:
        phi = rand() * 2 * pi
        min_phi = phi - 2 * pi
        max_phi = phi
    else:
        min_phi = - angle * rand()
        max_phi = min_phi + angle
        phi = min_phi + rand() * (max_phi - min_phi)

    # slice sampling loop
    while True:
        prop = xx * cos(phi) + nu * sin(phi)
        curlik = likfun(prop, params)
        if curlik > hh:
            break
        if phi > 0:
            max_phi = phi
        elif phi < 0:
            min_phi = phi
        else:
            raise IOError('BUG: slice sampling shrunk to the current position.')
        phi = min_phi + rand() * (max_phi - min_phi)

    return (prop, curlik)


if __name__ == "__main__":
    main()
