# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.linalg import cholesky, cho_solve
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
    plt.title('Gradient Decent', fontsize=20)

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
def kernel(x, x_prime, theta_1, theta_2, theta_3, noise, eval_grad=False):
    # delta function
    if noise:
        delta = theta_3
    else:
        delta = 0

    if eval_grad:
        dk_dTheta_1 = kernel(x, x_prime, theta_1, theta_2, theta_3, noise) - delta
        dk_dTheta_2 = (kernel(x, x_prime, theta_1, theta_2, theta_3, noise) - delta) * ((x - x_prime)**2) / theta_2
        dk_dTheta_3 = delta

        return rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + \
            delta, np.array([dk_dTheta_1, dk_dTheta_2, dk_dTheta_3])

    return rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + delta


def optimize(x_train, y_train, bounds, initial_params=np.ones(3)):
    bounds = np.atleast_2d(bounds)

    def log_marginal_likelihood(params):
        train_length = len(x_train)
        K = np.zeros((train_length, train_length))
        for x_idx in range(train_length):
            for x_prime_idx in range(train_length):
                K[x_idx, x_prime_idx] = kernel(
                    x_train[x_idx], x_train[x_prime_idx], params[0], params[1], params[2], x_idx == x_prime_idx)

        L_ = cholesky(K, lower=True)
        alpha_ = cho_solve((L_, True), y_train)
        return - 0.5 * np.dot(y_train.T, alpha_) - np.sum(np.log(np.diag(L_))) - 0.5 * train_length * np.log(2 * np.pi)

    def log_likelihood_gradient(params):
        train_length = len(x_train)
        K = np.zeros((train_length, train_length))
        dK_dTheta = np.zeros((3, train_length, train_length))
        # dK_dTheta = np.zeros((train_length, train_length, 3))
        for x_idx in range(train_length):
            for x_prime_idx in range(train_length):
                k, grad = kernel(x_train[x_idx], x_train[x_prime_idx], params[0],
                                 params[1], params[2], x_idx == x_prime_idx, eval_grad=True)
                K[x_idx, x_prime_idx] = k
                dK_dTheta[0, x_idx, x_prime_idx] = grad[0]
                dK_dTheta[1, x_idx, x_prime_idx] = grad[1]
                dK_dTheta[2, x_idx, x_prime_idx] = grad[2]
                # dK_dTheta[x_idx, x_prime_idx, 0] = grad[0]
                # dK_dTheta[x_idx, x_prime_idx, 1] = grad[1]
                # dK_dTheta[x_idx, x_prime_idx, 2] = grad[2]

        L_ = cholesky(K, lower=True)
        alpha_ = cho_solve((L_, True), y_train)
        K_inv = cho_solve((L_, True), np.eye(K.shape[0]))
        inner_term = np.einsum("i,j->ij", alpha_, alpha_) - K_inv
        # _ = np.einsum("ij,ijk->ijk", inner_term, dK_dTheta)
        tr = np.trace(np.array([np.dot(inner_term, dK_dTheta[0, :, :]), np.dot(
            inner_term, dK_dTheta[1, :, :]), np.dot(inner_term, dK_dTheta[2, :, :])]), axis1=1, axis2=2)

        # tr = np.trace(_)
        return 0.5 * tr

    def obj_func(params):
        lml = log_marginal_likelihood(params)
        grad = log_likelihood_gradient(params)
        return -lml, -grad

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

    L_ = cholesky(K, lower=True)
    alpha_ = cho_solve((L_, True), y_train)

    for x_test_idx in range(test_length):
        k = np.zeros((train_length,))
        for x_idx in range(train_length):
            k[x_idx] = kernel(
                x_train[x_idx],
                x_test[x_test_idx], thetas[0], thetas[1], thetas[2], x_idx == x_test_idx)

        s = kernel(
            x_test[x_test_idx],
            x_test[x_test_idx], thetas[0], thetas[1], thetas[2], x_test_idx == x_test_idx)
        v_ = cho_solve((L_, True), k.T)
        mu.append(np.dot(k, alpha_))
        var.append(s - np.dot(k, v_))
    return np.array(mu), np.array(var)


if __name__ == "__main__":
    mu, var = gpr(x_train, y_train, x_test)
    plot_gpr(x_train, y_train, x_test, mu, var)
