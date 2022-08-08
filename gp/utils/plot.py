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
    plt.savefig(f'{title}.png')
