from matplotlib import pyplot as plt
import numpy as np


def plot_gpr(x, y, x_train, y_train, x_test, mu, var):
    plt.figure(figsize=(16, 8))
    plt.title('Gradient Decent', fontsize=20)

    plt.plot(x, y, label='objective')
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
