# -*- coding: utf-8 -*-
import numpy as np

np.random.seed(42)


def elliptical(f, log_likelihood, L):
    nu = np.dot(L, np.random.randn(len(f)))
    rho = log_likelihood(f) + np.log(np.random.uniform(0, 1))
    theta = np.random.uniform(0, 2 * np.pi)
    st, ed = theta - 2 * np.pi, theta

    while True:
        print(theta)
        f = f * np.cos(theta) + nu * np.sin(theta)

        if log_likelihood(f) > rho:
            return f, log_likelihood(f)
        else:
            if theta > 0:
                ed = theta
            else:
                st = theta
            theta = np.random.uniform(st, ed)
