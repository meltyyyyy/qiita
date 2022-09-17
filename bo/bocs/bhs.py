import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


# Bayesian Horseshoe Sampler
# References:
# A simple sampler for the horseshoe estimator
# https://arxiv.org/pdf/1508.03884.pdf
def bhs(X, y, n_samples=1000, burnin=200):
    n, p = X.shape
    XtX = X.T @ X

    beta = np.zeros((p, n_samples))
    sigma2 = 1
    lambda2 = np.random.uniform(size=p)
    tau2 = 1
    nu = np.ones(p)
    xi = 1

    # Run Gibbs Sampler
    for i in range(n_samples + burnin):
        Lambda_star = tau2 * np.diag(lambda2)
        A = XtX + np.linalg.inv(Lambda_star)
        A_inv = np.linalg.inv(A)
        b = np.random.multivariate_normal(A_inv @ X.T @ y, sigma2 * A_inv)

        # Sample sigma^2
        e = y - np.dot(X, b)
        shape = (n + p) / 2.
        scale = np.dot(e.T, e) / 2. + np.sum(b**2 / lambda2) / tau2 / 2.
        sigma2 = 1. / np.random.gamma(shape, 1. / scale)

        # Sample lambda^2
        scale = 1. / nu + b**2. / 2. / tau2 / sigma2
        lambda2 = 1. / np.random.exponential(1. / scale)

        # Sample tau^2
        shape = (p + 1.) / 2.
        scale = 1. / xi + np.sum(b**2. / lambda2) / 2. / sigma2
        tau2 = 1. / np.random.gamma(shape, 1. / scale)

        # Sample nu
        scale = 1. + 1. / lambda2
        nu = 1. / np.random.exponential(1. / scale)

        # Sample xi
        scale = 1. + 1. / tau2
        xi = 1. / np.random.exponential(1. / scale)

        if i >= burnin:
            beta[:, i - burnin] = b

    return beta


if __name__ == '__main__':
    n = 20
    p = 20

    # true coefficients
    coefs = np.zeros((p,))
    coefs[0] = 1.0
    coefs[1] = 1.5
    coefs[2] = 0.5

    X = np.random.multivariate_normal(np.zeros((p,)), np.eye(p), size=n)
    y = X @ coefs

    beta = bhs(X, y)

    # traceplot
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    for i in range(3):
        for j in range(3):
            index = 3 * i + j
            if i == 0:
                axes[i][j].set_ylim(-5, 5)
            axes[i][j].set_title("beta_{}".format(index + 1))
            axes[i][j].plot(beta[index, :])
    fig.tight_layout()
    fig.savefig('bhs-tp.png')
    plt.close()

    # mean, variance
    fig = plt.figure(figsize=(16, 8))
    plt.xlabel('coefficients')
    plt.ylabel('value')
    for i in range(p):
        mean = np.mean(beta[i, :])
        var = np.var(beta[i, :])
        std = np.sqrt(np.abs(var))
        plt.errorbar(i, mean, yerr=std * 4, capsize=5, fmt='o', markersize=10,)
    fig.savefig('bhs-ci.png')
    plt.close()
