# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


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



n = 100
data_x = np.linspace(0, 4 * np.pi, n)
data_y = objective(data_x)


x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.90)


# Radiant Basis Kernel
def kernel(x, x_prime):
    return rbf(x, x_prime, theta_1=1.0, theta_2=1.0)


mu, var = gpr(x_train, y_train, x_test, kernel)


# len(x_train) is 10
ucb = UCB(mu, var, 10)
# pi = PI(mu, var)
# ei = EI(mu, var)


def plot_aquisition(mu, var, acqui):
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
