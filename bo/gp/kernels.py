# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def plot_kernel(kernel, title):
    n = 100
    x = np.linspace(-10, 10, n)
    ys = []

    for i in range(3):
        mkernel = np.zeros((x.shape[0], x.shape[0]))
        for i_row in range(x.shape[0]):
            for i_col in range(i_row, x.shape[0]):
                mkernel[i_row, i_col] = kernel(x[i_row], x[i_col])
                mkernel[i_col, i_row] = mkernel[i_row, i_col]

        K = 1**2 * np.dot(mkernel, mkernel.T)
        y = np.random.multivariate_normal(np.zeros(len(x)), K)
        ys.append(y)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for y in ys:
        ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title}')


def rbf(x, x_prime, theta_1, theta_2):
    """RBF Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return theta_1 * np.exp(-1 * (x - x_prime)**2 / theta_2)


def periodic(x, x_prime, theta_1, theta_2):
    """Periodic Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta_1 (float): hyper parameter
        theta_2 (float): hyper parameter
    """

    return np.exp(theta_1 * np.cos((x - x_prime) / theta_2))


def linear(x, x_prime, theta):
    """Linear Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta (float): hyper parameter
    """

    return x * x_prime + theta


def exp(x, x_prime, theta):
    """Exponential Kernel

    Args:
        x (float): data
        x_prime (float): data
        theta (float): hyper parameter
    """

    return np.exp(-np.abs(x - x_prime) / theta)


if __name__ == "__main__":
    plot_kernel(lambda x, x_prime: rbf(x, x_prime, 0.5, 0.5), 'RBF Kernel')
    plot_kernel(lambda x, x_prime: periodic(x, x_prime, 0.5, 0.5), 'Periodic Kernel')
    plot_kernel(lambda x, x_prime: linear(x, x_prime, 0.5), 'Linear Kernel')
    plot_kernel(lambda x, x_prime: rbf(x, x_prime, 0.5, 0.5), 'Exponential Kernel')
    plot_kernel(lambda x, x_prime: 0.8 * exp(x, x_prime, 0.5) + 0.2 * periodic(x, x_prime, 0.5, 0.5), 'Exponential + Periodic Kernel')
