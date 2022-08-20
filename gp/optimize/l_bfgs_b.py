# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
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


# Radiant Basis Kernel + Error
def kernel(x, x_prime, theta_1, theta_2, theta_3, noise, eval_grad=False):
    # delta function
    if noise:
        delta = theta_3
    else:
        delta = 0

    k = rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + delta

    if eval_grad:
        dk_dTheta_1 = kernel(x, x_prime, theta_1, theta_2, theta_3, noise) - delta
        dk_dTheta_2 = (kernel(x, x_prime, theta_1, theta_2, theta_3, noise) - delta) * ((x - x_prime)**2) / theta_2
        dk_dTheta_3 = delta

        return k, np.array([dk_dTheta_1, dk_dTheta_2, dk_dTheta_3])

    return k


def optimize(x_train, y_train, bounds, initial_params=np.ones(3)):
    bounds = np.atleast_2d(bounds)

    def log_marginal_likelihood(params):
        train_length = len(x_train)
        K = np.zeros((train_length, train_length))
        for x_idx in range(train_length):
            for x_prime_idx in range(train_length):
                K[x_idx, x_prime_idx] = kernel(
                    x_train[x_idx], x_train[x_prime_idx], params[0], params[1], params[2], x_idx == x_prime_idx)

        yy = np.dot(np.linalg.inv(K), y_train)
        return - (np.linalg.slogdet(K)[1] + np.dot(y_train, yy))

    def log_likelihood_gradient(params):
        train_length = len(x_train)
        K = np.zeros((train_length, train_length))
        dK_dTheta = np.zeros((3, train_length, train_length))
        for x_idx in range(train_length):
            for x_prime_idx in range(train_length):
                k, grad = kernel(x_train[x_idx], x_train[x_prime_idx], params[0],
                                 params[1], params[2], x_idx == x_prime_idx, eval_grad=True)
                K[x_idx, x_prime_idx] = k
                dK_dTheta[0, x_idx, x_prime_idx] = grad[0]
                dK_dTheta[1, x_idx, x_prime_idx] = grad[1]
                dK_dTheta[2, x_idx, x_prime_idx] = grad[2]

        K_inv = np.linalg.inv(K)
        yy = np.dot(K_inv, y_train)
        tr = np.einsum("ijj", np.einsum("ij,kjl->kil", K_inv, dK_dTheta))
        return - 0.5 * tr + 0.5 * np.einsum("i,ji->j", yy.T, np.einsum("ijk,k->ij", dK_dTheta, yy))

    def obj_func(params):
        _lml = log_marginal_likelihood(params)
        _grad = log_likelihood_gradient(params)
        return -_lml, -_grad

    opt_res = scipy.optimize.minimize(
        obj_func,
        initial_params,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
    )

    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min


def gpr(x_train, y_train, x_test):
    # average
    mu = []
    # variance
    var = []

    train_length = len(x_train)
    test_length = len(x_test)

    thetas, _ = optimize(x_train, y_train, bounds=np.array(
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
                x_test[x_test_idx], thetas[0], thetas[1], thetas[2], x_idx == x_test_idx)
        s = kernel(
            x_test[x_test_idx],
            x_test[x_test_idx], thetas[0], thetas[1], thetas[2], x_test_idx == x_test_idx)
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

    mu, var = gpr(x_train, y_train, x_test)
    plot_gpr(data_x, data_y, x_train, y_train, x_test, mu, var)
