# -*- coding: utf-8 -*-
"""Gaussian Process Regression
This is Bayesian Optimization implementation.
Bayesian optimization is particularly advantageous for problems where
f(x) is difficult to evaluate due to its computational cost.
"""


from aquisitions import UCB, PI, EI
from gpr import train_test_split, gpr
from kernels import rbf, periodic, exp, linear
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x)


n = 10000
data_x = np.linspace(0, 4 * np.pi, n)
data_y = objective(data_x)

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.9997)


def plot_bo(mu, var, trial, x_new, y_new):
    plt.figure(figsize=(24, 8))
    plt.title('Bayesian Optimization', fontsize=20)
    plt.xlim(-25, 25)
    plt.ylim(-120, 75)
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
        loc='upper right',
        fontsize=12)
    plt.savefig(f'bo_{trial}.png')
    plt.close()


n_trials = 15


def kernel(x, x_prime):
    return rbf(x, x_prime, theta_1=200.0, theta_2=1.0)


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
