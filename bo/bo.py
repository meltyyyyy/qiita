# -*- coding: utf-8 -*-
"""Gaussian Process Regression
This is Bayesian Optimization implementation.
Bayesian optimization is particularly advantageous for problems where
f(x) is difficult to evaluate due to its computational cost.
"""


import numpy as np
import matplotlib.pyplot as plt
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


def train_test_split(x, y, test_size):
    assert len(x) == len(y)
    n_samples = len(x)
    test_indices = np.sort(
        np.random.choice(
            np.arange(n_samples), int(
                n_samples * test_size), replace=False))
    train_indices = np.ones(n_samples, dtype=bool)
    train_indices[test_indices] = False
    test_indices = ~ train_indices

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


n = 10000
data_x = np.linspace(0, 4 * np.pi, n)
data_y = objective(data_x)

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.9997)


def plot_bo(mu, var, trial, x_new, y_new):
    plt.figure(figsize=(24, 8))
    plt.title('Bayesian Optimization', fontsize=20)
    plt.ylim(-6, 10)
    plt.plot(data_x, data_y, label='objective')

    std = np.sqrt(np.abs(var))
    plt.plot(data_x, mu, label='mean')
    plt.fill_between(
        data_x,
        mu + 2 * std,
        mu - 2 * std,
        color='turquoise',
        alpha=.2,
        label='standard deviation')
    plt.scatter(x_new, y_new, color='blue', label='next x')
    plt.legend(
        loc='lower left',
        fontsize=12)
    plt.savefig(f'bo_{trial}.png')
    plt.close()

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


n_trials = 15







def kernel(x, x_prime):
    return rbf(x, x_prime, theta_1=1.0, theta_2=1.0)


# Upper Confidence Bound
def UCB(mu, var):
    k = 20.0
    return mu + k * var


def bayes_opt(x_train, y_train, data_x, objective, n_trials, kernel):

    mu, var = gpr(x_train, y_train, data_x, kernel)

    for trial in range(n_trials):
        ucb = UCB(mu, var)
        arg_max = np.argmax(ucb)

        x_new = data_x[arg_max]
        y_new = objective(data_x[arg_max])

        if(x_new not in x_train):
            x_train = np.hstack([x_train, x_new])
            y_train = np.hstack([y_train, y_new])

        mu, var = gpr(x_train, y_train, data_x, kernel)
        plot_bo(mu, var, trial, x_new, y_new)


bayes_opt(x_train, y_train, data_x, objective, n_trials, kernel)
