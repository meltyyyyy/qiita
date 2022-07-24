# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from kernels import rbf, periodic, exp, linear
plt.style.use('seaborn-pastel')


def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x)


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


n = 100
data_x = np.linspace(0, 4 * np.pi, n)
data_y = objective(data_x)


x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.70)


def plot_gpr(x_train, y_train, x_test, mu, var):
    plt.figure(figsize=(16, 8))
    plt.title('MCMC', fontsize=20)

    plt.plot(data_x, data_y, label='objective')
    plt.plot(
        x_train,
        y_train,
        'o',
        label='train data')

    std = np.sqrt(np.abs(var))

    plt.plot(x_test, mu, label='mean')

    plt.fill_between(
        x_test,
        mu + 2 * std,
        mu - 2 * std,
        alpha=.2,
        label='standard deviation')
    plt.legend(
        loc='lower left',
        fontsize=12)

    plt.savefig('gpr.png')


# Radiant Basis Kernel + Error
def kernel(x, x_prime, theta_1, theta_2, theta_3, noise):
    # delta function
    if noise:
        delta = theta_3
    else:
        delta = 0

    return rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + delta


def optimize(x_train, y_train, bounds, initial_params=np.ones(3), n_iter=1000):
    params = initial_params
    bounds = np.atleast_2d(bounds)

    # log transformation
    log_params = np.log(params)
    log_bounds = np.log(bounds)
    log_scale = log_bounds[:, 1] - log_bounds[:, 0]

    def log_marginal_likelihood(params):
        train_length = len(x_train)
        K = np.zeros((train_length, train_length))
        for x_idx in range(train_length):
            for x_prime_idx in range(train_length):
                K[x_idx, x_prime_idx] = kernel(x_train[x_idx], x_train[x_prime_idx],
                                               params[0], params[1], params[2], x_idx == x_prime_idx)

        y = y_train
        yy = np.dot(np.linalg.inv(K), y_train)
        return - (np.linalg.slogdet(K)[1] + np.dot(y, yy))

    lml_prev = log_marginal_likelihood(params)

    thetas_list = []
    lml_list = []
    for _ in range(n_iter):
        move = 1e-2 * np.random.normal(0, log_scale, size=len(params))

        need_resample = (log_params +
                         move < log_bounds[:, 0]) | (log_params +
                                                     move > log_bounds[:, 1])

        while(np.any(need_resample)):
            move[need_resample] = np.random.normal(0, log_scale, size=len(params))[need_resample]
            need_resample = (log_params +
                             move < log_bounds[:, 0]) | (log_params +
                                                         move > log_bounds[:, 1])

        # proposed distribution
        next_log_params = log_params + move
        next_params = np.exp(next_log_params)
        lml_next = log_marginal_likelihood(next_params)

        r = np.exp(lml_next - lml_prev)

        # metropolis update
        if r > 1 or r > np.random.random():
            params = next_params
            log_params = next_log_params
            lml_prev = lml_next
            thetas_list.append(params)
            lml_list.append(lml_prev)

    return thetas_list[np.argmax(lml_list)]


def gpr(x_train, y_train, x_test):
    # average
    mu = []
    # variance
    var = []

    train_length = len(x_train)
    test_length = len(x_test)

    thetas = optimize(x_train, y_train, bounds=np.array(
        [[1e-2, 1e2], [1e-2, 1e2], [1e-2, 1e2]]), initial_params=np.array([0.5, 0.5, 0.5]))
    print(thetas)

    K = np.zeros((train_length, train_length))
    for x_idx in range(train_length):
        for x_prime_idx in range(train_length):
            K[x_idx, x_prime_idx] = kernel(
                x_train[x_idx], x_train[x_prime_idx], thetas[0], thetas[1], thetas[2], x_idx == x_prime_idx)

    yy = np.dot(np.linalg.inv(K), y_train)

    for x_test_idx in range(test_length):
        k = np.zeros((train_length,))
        for x_idx in range(train_length):
            k[x_idx] = kernel(
                x_train[x_idx],
                x_test[x_test_idx], thetas[0], thetas[1], thetas[2], x_idx == x_prime_idx)
        s = kernel(
            x_test[x_test_idx],
            x_test[x_test_idx], thetas[0], thetas[1], thetas[2], x_idx == x_prime_idx)
        mu.append(np.dot(k, yy))
        kK_ = np.dot(k, np.linalg.inv(K))
        var.append(s - np.dot(kK_, k.T))
    return np.array(mu), np.array(var)


if __name__ == "__main__":
    mu, var = gpr(x_train, y_train, x_test)
    plot_gpr(x_train, y_train, x_test, mu, var)
