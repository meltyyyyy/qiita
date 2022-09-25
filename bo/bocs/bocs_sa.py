import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from itertools import combinations

plt.style.use('seaborn-pastel')


class SparseBayesianLinearRegression:
    def __init__(self, n_vars: np.int64, order: np.int64, random_state: np.int64 = 42):
        assert n_vars > 0, "The number of variables must be greater than 0"
        self.n_vars = n_vars
        self.order = order
        self.rs = np.random.RandomState(random_state)
        self.n_coef = int(1 + n_vars + 0.5 * n_vars * (n_vars - 1))
        self.coefs = self.rs.normal(0, 1, size=self.n_coef)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Sparse Bayesian Linear Regression

        Args:
            X (np.ndarray): matrix of shape (n_samples, n_vars)
            y (np.ndarray): matrix of shape (n_samples, )
        """
        assert X.shape[1] == self.n_vars, "The number of variables does not match. X has {} variables, but n_vars is {}.".format(
            X.shape[1], self.n_vars)
        assert y.ndim == 1, "y should be 1 dimension of shape (n_samples, ), but is {}".format(
            y.ndim)

        # x_1, x_2, ... , x_n
        # ↓
        # x_1, x_2, ... , x_n, x_1*x_2, x_1*x_3, ... , x_n * x_ n-1
        X = self._order_effects(X)

        needs_sample = 1
        while (needs_sample):
            try:
                _coefs, _coef0 = self._bhs(X, y)
            except Exception as e:
                print(e)
                continue

            if not np.isnan(_coefs).any():
                needs_sample = 0

        self.coefs = np.append(_coef0, _coefs)

    def predict(self, x: np.ndarray) -> np.float64:
        assert x.shape[1] == self.n_vars, "The number of variables does not match. x has {} variables, but n_vars is {}.".format(
            x.shape[1], self.n_vars)
        x = self._order_effects(x)
        x = np.append(1, x)
        return x @ self.coefs

    def _order_effects(self, X: np.ndarray) -> np.ndarray:
        """Compute order effects
        Computes data matrix for all coupling
        orders to be added into linear regression model.

        Order is the number of combinations that needs to be taken into consideration,
        usually set to 2.

        Args:
            X (np.ndarray): input materix of shape (n_samples, n_vars)

        Returns:
            X_allpairs (np.ndarray): all combinations of variables up to consider,
                                     which shape is (n_samples, Σ[i=1, order] comb(n_vars, i))
        """
        assert X.shape[1] == self.n_vars, "The number of variables does not match. X has {} variables, but n_vars is {}.".format(
            X.shape[1], self.n_vars)

        n_samples, n_vars = X.shape
        X_allpairs = X.copy()

        for i in range(2, self.order + 1, 1):

            # generate all combinations of indices (without diagonals)
            offdProd = np.array(list(combinations(np.arange(n_vars), i)))

            # generate products of input variables
            x_comb = np.zeros((n_samples, offdProd.shape[0], i))
            for j in range(i):
                x_comb[:, :, j] = X[:, offdProd[:, j]]
            X_allpairs = np.append(X_allpairs, np.prod(x_comb, axis=2), axis=1)

        return X_allpairs

    def _bhs(self, X: np.ndarray, y: np.ndarray, n_samples: np.int64 = 1,
             burnin: np.int64 = 200) -> Tuple[np.ndarray, np.float64]:
        """Run Bayesian Horseshoe Sampler
        Sample coefficients from conditonal posterior using Gibbs Sampler
        <Reference>
        A simple sampler for the horseshoe estimator
        https://arxiv.org/pdf/1508.03884.pdf
        Args:
            X (np.ndarray): input materix of shape (n_samples, 1 + Σ[i=1, order] comb(n_vars, i)).
            y (np.ndarray): matrix of shape (n_samples, ).
            n_samples (np.int64, optional): The number of sample. Defaults to 1.
            burnin (np.int64, optional): The number of sample to be discarded. Defaults to 200.

        Returns:
            Union[np.ndarray, np.float64]: Coefficients for Linear Regression.
        """

        assert X.shape[1] == self.n_coef - 1, "The number of combinations is wrong, it should be {}".format(
            self.n_coef)
        assert y.ndim == 1, "y should be 1 dimension of shape (n_samples, ), but is {}".format(
            y.ndim)

        n, p = X.shape
        XtX = X.T @ X

        beta = np.zeros((p, n_samples))
        beta0 = np.mean(y)
        sigma2 = 1
        lambda2 = self.rs.uniform(size=p)
        tau2 = 1
        nu = np.ones(p)
        xi = 1

        # Run Gibbs Sampler
        for i in range(n_samples + burnin):
            Lambda_star = tau2 * np.diag(lambda2)
            A = XtX + np.linalg.inv(Lambda_star)
            A_inv = np.linalg.inv(A)
            b = self.rs.multivariate_normal(A_inv @ X.T @ y, sigma2 * A_inv)

            # Sample sigma^2
            e = y - np.dot(X, b)
            shape = (n + p) / 2.
            scale = np.dot(e.T, e) / 2. + np.sum(b**2 / lambda2) / tau2 / 2.
            sigma2 = 1. / self.rs.gamma(shape, 1. / scale)

            # Sample lambda^2
            scale = 1. / nu + b**2. / 2. / tau2 / sigma2
            lambda2 = 1. / self.rs.exponential(1. / scale)

            # Sample tau^2
            shape = (p + 1.) / 2.
            scale = 1. / xi + np.sum(b**2. / lambda2) / 2. / sigma2
            tau2 = 1. / self.rs.gamma(shape, 1. / scale)

            # Sample nu
            scale = 1. + 1. / lambda2
            nu = 1. / self.rs.exponential(1. / scale)

            # Sample xi
            scale = 1. + 1. / tau2
            xi = 1. / self.rs.exponential(1. / scale)

            if i >= burnin:
                beta[:, i - burnin] = b

        return beta, beta0

def simulated_annealinng(objective, n_vars: np.int64, cooling_rate: np.float64 = 0.985,
                         n_iter: np.int64 = 100) -> Union[np.ndarray, np.ndarray]:
    """Run simulated annealing
    Simulated Annealing (SA) is a probabilistic technique
    for approximating the global optimum of a given function.

    Args:
        objective : objective function / statistical model
        n_vars (np.int64): The number of variables
        cooling_rate (np.float64, optional): Defaults to 0.985.
        n_iter (np.int64, optional): Defaults to 100.

    Returns:
        Union[np.ndarray, np.ndarray]: Best solutions that maximize objective.
    """
    X = np.zeros((n_iter, n_vars))
    obj = np.zeros((n_iter, ))

    # set initial temperature and cooling schedule
    T = 1.
    def cool(T): return cooling_rate * T

    curr_x = sample_binary_matrix(1, n_vars)
    curr_obj = objective(curr_x)

    best_x = curr_x
    best_obj = curr_obj

    for i in range(n_iter):

        # decrease T according to cooling schedule
        T = cool(T)

        new_x = sample_binary_matrix(1, n_vars)
        new_obj = objective(new_x)

        # update current solution
        if (new_obj > curr_obj) or (rs.rand()
                                    < np.exp((new_obj - curr_obj) / T)):
            curr_x = new_x
            curr_obj = new_obj

        # Update best solution
        if new_obj > best_obj:
            best_x = new_x
            best_obj = new_obj

        # save solution
        X[i, :] = best_x
        obj[i] = best_obj

    return X, obj
