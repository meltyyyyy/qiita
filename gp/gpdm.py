# -*- coding: utf-8 -*-


import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import make_s_curve
from scipy.optimize import fmin_l_bfgs_b
from sklearn.decomposition import PCA
plt.style.use('seaborn-pastel')

r = np.random.RandomState(42)


def make_dataset(n_samples=200, n_dims=40):
    X, t = make_s_curve(n_samples, random_state=42)

    X = np.delete(X, obj=1, axis=1)

    indices = t.argsort()
    X, t = X[indices], t[indices]

    K = kernel(X, 1, 2)

    F = r.multivariate_normal(np.zeros(n_samples), K, size=n_dims).T
    Y = F + r.normal(0, scale=1, size=F.shape)
    return X, Y, t


def rbf(x, x_prime, theta_1, theta_2):
    """RBF Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return theta_1 * np.exp(-1 * np.sum((x - x_prime)**2) / theta_2)


def kernel(X, theta_1, theta_2):
    length = X.shape[0]

    K = np.zeros((length, length))
    for x_idx in range(length):
        for x_prime_idx in range(length):
            K[x_idx, x_prime_idx] += rbf(X[x_idx], X[x_prime_idx], theta_1, theta_2)

    return K


# Radiant Basis Kernel + Error
# def kernel_Y(X, theta_1, theta_2, theta_3):
#     length = X.shape[0]

#     K = np.zeros((length, length))
#     for x_idx in range(length):
#         for x_prime_idx in range(length):
#             try:
#                 k = rbf(X[x_idx], X[x_prime_idx], theta_1, theta_2)
#                 K[x_idx, x_prime_idx] += k
#             except ValueError:
#                 k = rbf(X._value[x_idx], X._value[x_prime_idx], theta_1, theta_2)
#                 K[x_idx, x_prime_idx] += k._value

#     K += theta_3 * np.eye(length)
#     return K


def kernel_Y(X, theta_1, theta_2, theta_3):
    length = X.shape[0]
    diffs = np.expand_dims(X / theta_2, 1) - np.expand_dims(X / theta_2, 0)
    return theta_1 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2)) + theta_3 * np.eye(length)


def kernel_X(X, theta_1, theta_2, theta_3, theta_4):
    K = kernel_Y(X, theta_1, theta_2, theta_3)
    return K + theta_4 * X @ X.T


def log_posterior(Y, X, beta, alpha):
    _, n_dims = Y.shape
    D = X.shape[1]

    K_Y = kernel_Y(X, *beta)
    det_term = -n_dims * np.prod(np.linalg.slogdet(K_Y)) / 2
    tr_term = -1 * np.trace(np.linalg.inv(K_Y) @ Y @ Y.T) / 2
    LY = det_term + tr_term

    K_X = kernel_X(X[:-1], *alpha)
    x = X[1:]
    det_term = - D * np.prod(np.linalg.slogdet(K_X)) / 2
    tr_term = -1 * np.trace(np.linalg.inv(K_X) @ x @ x.T) / 2
    LX = det_term + tr_term
    return - LY - LX


def optimize_gpdm(Y, n_components):
    n_samples = Y.shape[0]

    X0 = r.multivariate_normal(np.zeros(n_samples), np.eye(n_samples), size=n_components).T
    X0 = X
    beta0 = np.array([1, 1, 1e-6])
    alpha0 = np.array([1, 1, 1e-6, 1e-6])

    def _lml(params):
        X = params[:n_samples * n_components].reshape(X0.shape)
        beta = params[n_samples * n_components:n_samples * n_components + 3]
        alpha = params[n_samples * n_components + 3:]
        return log_posterior(Y, X, beta, alpha)

    def obj_func(params):
        _grad = grad(_lml)
        return _lml(params), _grad(params)

    x0 = np.concatenate([X0.flatten(), beta0, alpha0])
    opt_res = fmin_l_bfgs_b(obj_func, x0, epsilon=1e-32)
    X_map = opt_res[0][:n_samples * n_components].reshape(X0.shape)
    return X_map


if __name__ == "__main__":
    X, Y, t = make_dataset(n_samples=200)

    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap='Blues')
    fig.savefig("scurve.png")

    X_map = optimize_gpdm(Y, n_components=2)

    fig = plt.figure()
    plt.scatter(X_map[:, 0], X_map[:, 1], c=t, cmap='Blues')
    fig.savefig("gpdm.png")
