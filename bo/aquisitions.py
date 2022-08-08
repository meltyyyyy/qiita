# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from train_test_split import train_test_split

plt.style.use('seaborn-pastel')


def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x)


# Upper Confidence Bound
def UCB(mu, var, trial):
    eps = 1e-3
    k = np.sqrt(np.abs(np.log(trial + eps)) / (trial + eps))
    return mu + k * var


# Probability of Improvement
def PI(mean, var):
    eps = 1e-7
    y_hat = np.max(mu)
    theta = (mean - y_hat) / (var + eps)
    return np.array([norm.cdf(theta[i]) for i in range(len(theta))])


# Expected Improvement
def EI(mu, var):
    eps = 1e-7
    sigma = np.sqrt(np.abs(var))
    y_hat = np.max(mu)
    theta = (mu - y_hat) / (sigma + eps)
    return np.array([(mu[i] - y_hat) * norm.cdf(theta[i]) +
                    sigma[i] * norm.pdf(theta[i]) for i in range(len(theta))])


def rbf(x, x_prime, theta_1, theta_2):
    """RBF Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return theta_1 * np.exp(-1 * (x - x_prime)**2 / theta_2)


def kernel(x, x_prime):
    return rbf(x, x_prime, theta_1=1.0, theta_2=1.0)


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


def plot_aquisition(data_x, data_y, mu, var, acqui):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title('Gaussian Process Regression', fontsize=20)

    plt.plot(data_x, data_y, label='objective')
    std = np.sqrt(np.abs(var))

    plt.plot(data_x, mu, label='mean')
    plt.fill_between(
        data_x,
        mu + 2 * std,
        mu - 2 * std,
        alpha=.2,
        label='standard deviation')

    plt.legend(
        loc='lower left',
        fontsize=12)
    plt.subplot(2, 1, 2)
    plt.title('Expected Improvement', fontsize=20)
    plt.plot(
        data_x,
        acqui,
        label='expected improvement')
    index = np.argmax(acqui)
    plt.scatter(data_x[index], acqui[index], color='blue', label='next x')
    plt.legend(
        loc='lower left',
        fontsize=12)
    plt.tight_layout()
    plt.savefig('acquisition.png')


if __name__ == "__main__":
    n = 100
    data_x = np.linspace(0, 4 * np.pi, n)
    data_y = objective(data_x)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.90)

    mu, var = gpr(x_train, y_train, x_test, kernel)

    # len(x_train) is 10
    ucb = UCB(mu, var, 10)
    # pi = PI(mu, var)
    # ei = EI(mu, var)

    plot_aquisition(data_x, data_y, mu, var, ucb)
