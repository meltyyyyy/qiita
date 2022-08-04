# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
from GPy.models import GPRegression
from GPy.kern import RBF
from GPy.core.gp import GP
from GPy.likelihoods import StudentT
from GPy.inference.latent_function_inference import Laplace
from utils.train_test_split import train_test_split

plt.style.use('seaborn-pastel')

np.random.seed(42)


def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x) + 5 * \
        np.random.binomial(1, 0.05, len(x)).reshape(len(x), 1)


def plot_gpr(x, y, x_train, y_train, x_test, mu, var, model, title):
    f_samples = model.posterior_samples_f(x_test.reshape((-1, 1)), size=10).reshape(-1, 10)

    # convert (n,1) -> (n,)
    x = x.reshape(-1)
    y = y.reshape(-1)
    x_train = x_train.reshape(-1)
    x_test = x_test.reshape(-1)
    y_train = y_train.reshape(-1)
    mu = mu.reshape(-1)
    var = var.reshape(-1)

    plt.figure(figsize=(16, 8))
    plt.title('t-Distribution', fontsize=20)
    plt.plot(x, y, 'o', label='objective')
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
    # plt.plot(x_test.reshape(-1), f_samples)
    plt.legend(
        loc='lower left',
        fontsize=12)

    plt.savefig(f'{title}')


if __name__ == "__main__":
    n = 200
    data_x = np.linspace(0, 4 * np.pi, n).reshape(n, 1)
    data_y = objective(data_x)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

    kernel = RBF(input_dim=1, ARD=True)
    gpr = GPRegression(x_train, y_train, kernel)
    gpr.optimize()
    mu, var = gpr.predict(x_test)

    plot_gpr(data_x, data_y, x_train, y_train, x_test, mu, var, gpr, "gpr.png")

    t_dist = StudentT(deg_free=7, sigma2=2)
    laplace = Laplace()
    tgpr = GP(x_train, y_train, kernel=kernel, likelihood=t_dist, inference_method=laplace)
    tgpr.optimize()
    mu, var = gpr.predict(x_test)

    plot_gpr(data_x, data_y, x_train, y_train, x_test, mu, var, gpr, "t_dist.png")
