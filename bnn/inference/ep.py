import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

r = np.random.RandomState(42)


def p(x):
    return np.exp(- x ** 2 / 2) * sigmoid(20 * x + 4)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def expectation_propagtion(n_samples=200, n_iter=100):
    mu = np.zeros((n_iter,))
    var = np.ones((n_iter, ))

    mu_prior = 0
    var_prior = 1

    for i in range(n_samples):
        for n in range(n_iter):
            var_next = 1/(1 / var_prior - 1 / var[n])
            mu_next = mu_prior + var_next * (mu_prior - mu[n]) / var[n]
            



if __name__ == '__main__':
    n = 100
    x = np.linspace(-2.5, 5.0, num=n)
    y = p(x)
    fig = plt.figure(figsize=(16, 8))
    plt.plot(x, y)
    fig.savefig('target.png')
