# -*- coding: utf-8 -*-


import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import make_s_curve
from scipy.optimize import fmin_l_bfgs_b
from sklearn.decomposition import PCA
plt.style.use('seaborn-pastel')


def make_dataset(n_samples=200, n_dims=40):
    r = np.random.RandomState(42)
    X, t = make_s_curve(n_samples, random_state=42)

    X = np.delete(X, obj=1, axis=1)

    indices = t.argsort()
    X, t = X[indices], t[indices]

    K = np.zeros((n_samples, n_samples))
    for x_idx in range(n_samples):
        for x_prime_idx in range(n_samples):
            K[x_idx, x_prime_idx] = rbf(X[x_idx], X[x_prime_idx], 1, 2)

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


# Radiant Basis Kernel + Error
def kernel_Y(x, x_prime, theta_1, theta_2, theta_3, noise):
    # delta function
    if noise:
        delta = theta_3
    else:
        delta = 0

    return rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + delta

# def _kernel_Y(X, theta_1, theta_2, theta_3):


def rbf_kernel(X, var, length_scale, diag):
    N = X.shape[0]
    diffs = np.expand_dims(X / length_scale, 1) - \
        np.expand_dims(X / length_scale, 0)
    return var * np.exp(-0.5 * np.sum(diffs ** 2, axis=2)) + diag * np.eye(N)


def log_posterior(Y, X, beta, alpha):
    _, n_dims = Y.shape
    D = X.shape[1]

    K_Y = rbf_kernel(X, *beta)
    det_term = -n_dims / 2 * np.prod(np.linalg.slogdet(K_Y))
    tr_term = -1 / 2 * np.trace(np.linalg.inv(K_Y) @ Y @ Y.T)
    LL = det_term + tr_term

    K_X = rbf_linear_kernel(X[:-1], *alpha)
    X_bar = X[1:]
    det_term = - D / 2 * np.prod(np.linalg.slogdet(K_X))
    tr_term = -1 / 2 * np.trace(np.linalg.inv(K_X) @ X_bar @ X_bar.T)
    LP = det_term + tr_term

    return LL + LP


def rbf_linear_kernel(X, var, length_scale, diag1, diag2):
    rbf = rbf_kernel(X, length_scale, var, diag1)
    linear = diag2 * X @ X.T
    return rbf + linear


def optimize_gpdm(Y, X0):
    T, D = X0.shape

    beta0 = np.array([1, 1, 1e-6])
    alpha0 = np.array([1, 1, 1e-6, 1e-6])

    def _neg_f(params):
        X = params[:T * D].reshape(X0.shape)
        beta = params[T * D:T * D + 3]
        alpha = params[T * D + 3:]
        return -1 * log_posterior(Y, X, beta, alpha)

    _neg_fp = grad(_neg_f)

    def f_fp(params):
        return _neg_f(params), _neg_fp(params)

    x0 = np.concatenate([X0.flatten(), beta0, alpha0])
    res = fmin_l_bfgs_b(f_fp, x0)
    X_map = res[0][:T * D].reshape(X0.shape)

    return X_map


if __name__ == "__main__":
    X, Y, t = make_dataset()

    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=t, cmap='Blues')
    # fig.savefig("scurve.png")

    X_map = optimize_gpdm(Y, X)

    fig = plt.figure()
    plt.scatter(X_map[:, 0], X_map[:, 1], c=t, cmap='Blues')
    fig.savefig("gpdm.png")
