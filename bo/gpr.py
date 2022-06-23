# -*- coding: utf-8 -*-
"""Gaussian Process Regression
This is Gaussian Process Regression implementation.
Gaussian Process is a stochastic process,
such that every finite collection of those random variables has a multivariate normal distribution.
"""

import numpy as np
from matplotlib import pyplot as plt


def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * \
        np.sin(2 / 3 * x) + np.random.randn(len(x))


n = 100
data_x = np.linspace(0, 4 * np.pi, n)
data_y = objective(data_x)


missing_rate = 0.2
sample_idx = np.sort(
    np.random.choice(
        np.arange(n), int(
            n * missing_rate), replace=False))

plt.figure(figsize=(12, 5))
plt.title('signal data', fontsize=20)

# original signals
plt.plot(data_x, data_y, 'x', color='green', label='correct signal')

# sample signal
plt.plot(
    data_x[sample_idx],
    data_y[sample_idx],
    'o',
    color='red',
    label='sample dots')

plt.legend(
    bbox_to_anchor=(
        1.05,
        1),
    loc='upper left',
    borderaxespad=0,
    fontsize=12)
plt.savefig('signal.png')


def kernel(x, x_prime, p, q, r):
    """Kernel Function

    Args:
        x (float): data
        x_prime (float): data
        p (float): hyper parameter
        q (float): hyper parameter
        r (float): error

    """
    if x == x_prime:
        delta = 1
    else:
        delta = 0

    return p * np.exp(-1 * (x - x_prime)**2 / q) + (r * delta)


# training data
x_train = np.copy(data_x[sample_idx])
y_train = np.copy(data_y[sample_idx])

# test data
x_test = np.copy(data_x)

# average
mu = []
# variance
var = []

# hyper parameters
Theta_1 = 1.0
Theta_2 = 0.4
Theta_3 = 0.1

train_length = len(x_train)
test_length = len(x_test)


K = np.zeros((train_length, train_length))
for x_idx in range(train_length):
    for x_prime_idx in range(train_length):
        K[x_idx, x_prime_idx] = kernel(
            x=x_idx, x_prime=x_prime_idx, p=Theta_1, q=Theta_2, r=Theta_3)

yy = np.dot(np.linalg.inv(K), y_train)

for x_test_idx in range(test_length):
    k = np.zeros((train_length,))

    for x_idx in range(train_length):
        k[x_idx] = kernel(
            x=x_train[x_idx],
            x_prime=x_test[x_test_idx],
            p=Theta_1,
            q=Theta_2,
            r=Theta_3)

    s = kernel(
        x=x_test[x_test_idx],
        x_prime=x_test[x_test_idx],
        p=Theta_1,
        q=Theta_2,
        r=Theta_3)
    mu.append(np.dot(k, yy))
    kK_ = np.dot(k, np.linalg.inv(K))
    var.append(s - np.dot(kK_, k.T))

plt.figure(figsize=(16, 8))
plt.title('Signal prediction by Gaussian process', fontsize=20)

plt.plot(data_x, data_y, 'x', color='green', label='correct signal')
plt.plot(
    data_x[sample_idx],
    data_y[sample_idx],
    'o',
    color='red',
    label='sample dots')

std = np.sqrt(np.abs(var))

plt.plot(x_test, mu, color='red', label='Mean by Gaussian process')
plt.fill_between(
    x_test,
    mu + 2 * std,
    mu - 2 * std,
    alpha=.2,
    color='pink',
    label='Standard deviation by Gaussian process')

plt.legend(
    bbox_to_anchor=(
        1.05,
        1),
    loc='upper left',
    borderaxespad=0,
    fontsize=12)
plt.savefig('gpr.png')
