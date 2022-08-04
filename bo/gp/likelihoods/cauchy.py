# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.style.use('seaborn-pastel')


def objective(x):
    r = np.random.RandomState(42)
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x) + 10 * \
        r.binomial(1, 0.05, len(x))


def rbf(x, x_prime, theta_1, theta_2):
    """RBF Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return theta_1 * np.exp(-1 * (x - x_prime)**2 / theta_2)


# Radiant Basis Kernel
def kernel(x, x_prime, noise):
    theta_3 = 1e-6
    # delta function
    if noise:
        delta = theta_3
    else:
        delta = 0
    return rbf(x, x_prime, theta_1=1.0, theta_2=1.0) + delta


def plot_gpr(x, y, f_posterior):
    plt.figure(figsize=(16, 8))
    plt.title('Cauchy', fontsize=20)
    plt.plot(x, y, 'x', label='objective')

    for i in range(f_posterior.shape[1]):
        plt.plot(x, f_posterior[:, i])
    plt.savefig('cauchy.png')


def gpr(x, y, kernel, n_iter=100):
    N = len(x)

    K = np.zeros((N, N))
    for x_idx in range(N):
        for x_prime_idx in range(N):
            K[x_idx, x_prime_idx] = kernel(
                x[x_idx], x[x_prime_idx], x_idx == x_prime_idx)

    K_inv = np.linalg.inv(K)
    L_ = np.linalg.cholesky(K)

    # Normalization
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std

    def log_marginal_likelihood(y, f, gamma=0.2):
        cauchy = - np.sum(np.log(gamma + (y - f)**2 / gamma))
        normal = - 0.5 * np.dot(f, np.dot(K_inv, f))
        return cauchy + normal

    burn_in = 50
    n_samples = n_iter - burn_in
    assert n_iter > burn_in

    f = np.dot(L_, np.random.randn(N))
    f_posterior = np.zeros((f.shape[0], n_samples))
    for i in tqdm(range(n_iter)):

        sampling = True
        while sampling:
            try:
                f, _ = elliptical(f, lambda f: log_marginal_likelihood(y, f), L_)
                sampling = False
            except IOError:
                sampling = True
                # print('Slice sampling shrunk to the current position. Retry sampling ...')

        if i >= burn_in:
            f_posterior[:, i - burn_in] = f * y_std + y_mean

    return f_posterior


def elliptical(f, log_likelihood, L):
    """elipitical sampling

    f is Gaussian Process
    f ~ N(0, K)

    Args:
        f : target distribution
        log_likelihood : log likelihood of f
        L : triangle matrix of K

    """
    rho = log_likelihood(f) + np.log(np.random.uniform(0, 1))
    nu = np.dot(L, np.random.randn(len(f)))

    theta = np.random.uniform(0, 2 * np.pi)
    st, ed = theta - 2 * np.pi, theta

    while True:
        f = f * np.cos(theta) + nu * np.sin(theta)
        if log_likelihood(f) > rho:
            return f, log_likelihood(f)
        else:
            if theta > 0:
                ed = theta
            elif theta < 0:
                st = theta
            else:
                raise IOError('Slice sampling shrunk to the current position.')
            theta = np.random.uniform(st, ed)


if __name__ == "__main__":
    n = 100
    x = np.linspace(0, 4 * np.pi, n)
    y = objective(x)

    f_posterior = gpr(x, y, kernel)

    plot_gpr(x, y, f_posterior)

    # K_MM = np.zeros((test_length, test_length))
    # for x_idx in range(test_length):
    #     for x_prime_idx in range(test_length):
    #         K_MM[x_idx, x_prime_idx] = kernel(
    #             x_test[x_idx], x_test[x_prime_idx], x_idx == x_prime_idx)

    # K_NM = np.zeros((train_length, test_length))
    # for x_idx in range(train_length):
    #     for x_prime_idx in range(test_length):
    #         K_NM[x_idx, x_prime_idx] = kernel(
    #             x_train[x_idx], x_test[x_prime_idx], noise=False)

    # n_fs = 10
    # f_samples = np.zeros((test_length, n_fs))
    # for i in range(n_fs):
    #     sample_idx = np.random.randint(n_samples)
    #     f_n = f_posterior[:, sample_idx]
    #     _ = np.einsum("ij,jk,k->i", K_NM.T, K_inv, f_n)
    #     var = K_MM - np.einsum("ij,jk,kl->il", K_NM.T, K_inv, K_NM)
    #     L_ = np.linalg.cholesky(var)
    #     f_m = np.dot(L_, np.random.randn(test_length))
    #     f_samples[:, i] = f_m