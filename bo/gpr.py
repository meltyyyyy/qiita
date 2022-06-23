# -*- coding: utf-8 -*-
"""Gaussian Process Regression
This is Gaussian Process Regression implementation.
Gaussian Process is a stochastic process,
such that every finite collection of those random variables has a multivariate normal distribution
"""

import numpy as np
from matplotlib import pyplot as plt

DATA_FIG = 'signal.png'
OUTPUT_FIG = 'gpr.png'

# original data
n = 100
data_x = np.linspace(0, 4 * np.pi, n)
data_y = 2 * np.sin(data_x) + 3 * np.cos(2 * data_x) + 5 * \
    np.sin(2 / 3 * data_x) + np.random.randn(len(data_x))

# make sample data
missing_value_rate = 0.2
sample_index = np.sort(
    np.random.choice(
        np.arange(n), int(
            n * missing_value_rate), replace=False))

plt.figure(figsize=(12, 5))
plt.title('signal data', fontsize=20)

# original signals
plt.plot(data_x, data_y, 'x', color='green', label='correct signal')

# sample signal
plt.plot(
    data_x[sample_index],
    data_y[sample_index],
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
plt.savefig(DATA_FIG)


def kernel(x, x_prime, p, q, r):
    """Radial Basis Function

    Args:
        x (_type_): _description_
        x_prime (_type_): _description_
        p (_type_): hyper parameter
        q (_type_): hyper parameter
        r (_type_): error

    Returns:
        _type_: _description_
    """
    if x == x_prime:
        delta = 1
    else:
        delta = 0

    return p * np.exp(-1 * (x - x_prime)**2 / q) + (r * delta)


# training data
x_train = np.copy(data_x[sample_index])
y_train = np.copy(data_y[sample_index])

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
plt.title('signal prediction by Gaussian process', fontsize=20)

plt.plot(data_x, data_y, 'x', color='green', label='correct signal')
plt.plot(
    data_x[sample_index],
    data_y[sample_index],
    'o',
    color='red',
    label='sample dots')

std = np.sqrt(np.abs(var))

plt.plot(x_test, mu, color='blue', label='mean by Gaussian process')
plt.fill_between(
    x_test,
    mu + 2 * std,
    mu - 2 * std,
    alpha=.2,
    color='blue',
    label='standard deviation by Gaussian process')

plt.legend(
    bbox_to_anchor=(
        1.05,
        1),
    loc='upper left',
    borderaxespad=0,
    fontsize=12)
plt.savefig(OUTPUT_FIG)
