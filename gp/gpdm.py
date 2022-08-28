# -*- coding: utf-8 -*-


import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.datasets import make_s_curve
from scipy.optimize import fmin_l_bfgs_b
from sklearn.decomposition import PCA
from GPy.models import GPLVM
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


def kernel(X, theta_1, theta_2):
    length = X.shape[0]

    K = np.zeros((length, length))
    for x_idx in range(length):
        for x_prime_idx in range(length):
            K[x_idx, x_prime_idx] += rbf(X[x_idx], X[x_prime_idx], theta_1, theta_2)

    return K


def rbf(x, x_prime, theta_1, theta_2):
    """RBF Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return theta_1 * np.exp(-1 * np.sum((x - x_prime)**2) / theta_2)


# # Radiant Basis Kernel + Error
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

    pca = PCA(n_components=n_components)
    X_init = pca.fit_transform(Y)
    beta0 = np.array([1, 1, 1e-6])
    alpha0 = np.array([1, 1, 1e-6, 1e-6])

    def _lml(params):
        X = params[:n_samples * n_components].reshape(X_init.shape)
        beta = params[n_samples * n_components:n_samples * n_components + 3]
        alpha = params[n_samples * n_components + 3:]
        return log_posterior(Y, X, beta, alpha)

    def obj_func(params):
        _grad = grad(_lml)
        return _lml(params), _grad(params)

    x0 = np.concatenate([X_init.flatten(), beta0, alpha0])
    opt_res = fmin_l_bfgs_b(obj_func, x0, epsilon=1e-32)
    X_map = opt_res[0][:n_samples * n_components].reshape(X_init.shape)
    return X_map


def plot(X_map, title):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"{title}", fontsize=18, fontweight='bold')

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Scatter", fontsize=18)
    ax1.scatter(X_map[:, 0], X_map[:, 1], c=t, cmap='Blues')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Line", fontsize=18)
    ax2.plot(X_map[:, 0], X_map[:, 1])

    fig.tight_layout()
    fig.savefig("{}.png".format(title.lower()))
    plt.close()


if __name__ == "__main__":
    X, Y, t = make_dataset(n_samples=200)

    # original data
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap='Blues')
    fig.savefig("scurve.png")

    # pca
    pca = PCA(n_components=2)
    X_map = pca.fit_transform(Y)
    plot(X_map, title='PCA')

    # gplvm
    gplvm = GPLVM(Y, input_dim=2)
    gplvm.optimize()
    X_map = gplvm.X
    plot(X_map, title='GPLVM')

    # gpdm
    X_map = optimize_gpdm(Y, n_components=2)
    plot(X_map, title='GPDM')
