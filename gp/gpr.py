# -*- coding: utf-8 -*-
"""Gaussian Process Regression
This is Gaussian Process Regression implementation.
Gaussian Process is a stochastic process,
such that every finite collection of those random variables has a multivariate normal distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.train_test_split import train_test_split
from utils.plot import plot_gpr
plt.style.use('seaborn-pastel')


def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x)


def rbf(x, x_prime, theta_1, theta_2):
    """RBF Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return theta_1 * np.exp(-1 * (x - x_prime)**2 / theta_2)


# Radiant Basis Kernel
def kernel(x, x_prime, theta_1=1.0, theta_2=1.0):

    return rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2)


def gpr(x_train, y_train, x_test, kernel):
    # average
    mu = []
    # variance
    var = []

    train_length = len(x_train)
    test_length = len(x_test)

    K = np.zeros((train_length, train_length))
    for x_idx in range(train_length):
        for x_prime_idx in range(train_length):
            K[x_idx, x_prime_idx] = kernel(
                x_train[x_idx], x_train[x_prime_idx])

    yy = np.dot(np.linalg.inv(K), y_train)

    for x_test_idx in range(test_length):
        k = np.zeros((train_length,))
        for x_idx in range(train_length):
            k[x_idx] = kernel(
                x_train[x_idx],
                x_test[x_test_idx])
        s = kernel(
            x_test[x_test_idx],
            x_test[x_test_idx])
        mu.append(np.dot(k, yy))
        kK_ = np.dot(k, np.linalg.inv(K))
        var.append(s - np.dot(kK_, k.T))
    return np.array(mu), np.array(var)


if __name__ == "__main__":
    n = 100
    data_x = np.linspace(0, 4 * np.pi, n)
    data_y = objective(data_x)

    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.70)

    mu, var = gpr(x_train, y_train, x_test, kernel)
    plot_gpr(data_x, data_y, x_train, y_train, x_test, mu, var)
