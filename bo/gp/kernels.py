# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from utils.plot import plot_kernel
plt.style.use('seaborn-pastel')


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
    plot_kernel(lambda x, x_prime: 0.8 * exp(x, x_prime, 0.5) + 0.2 *
                periodic(x, x_prime, 0.5, 0.5), 'Exponential + Periodic Kernel')
