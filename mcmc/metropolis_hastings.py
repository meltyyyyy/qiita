# -*- coding: utf-8 -*-
"""Metropolis-Hastings methods
This is Metropolis-Hastings method implementation.
Hamiltonian Monte Carlo corresponds to an instance of the Metropolis-Hastings algorithm,
with a Hamiltonian dynamics evolution simulated
using a time-reversible and volume-preserving numerical integrator to propose a move
to a new point in the state space.
"""

import copy

import numpy as np
import matplotlib.pyplot as plt


# P(x) : Target distribution
def P(x1, x2, a):
    return np.exp(-0.5 * (x1**2 - 2 * a * x1 * x2 + x2**2))


# Q(x) : Proposal distribution
def Q(c, mu1, mu2, sigma):
    return (
        c[0] +
        np.random.normal(
            mu1,
            sigma),
        c[1] +
        np.random.normal(
            mu2,
            sigma))


def metropolis(N, mu1, mu2, sigma, b):
    current = (10, 10)
    sample = []
    sample.append(current)
    accept_ratio = []

    for i in range(N):
        candidate = Q(current, mu1, mu2, sigma)

        T_prev = P(current[0], current[1], b)
        T_next = P(candidate[0], candidate[1], b)
        a = T_next / T_prev

        if a > 1 or a > np.random.uniform(0, 1):
            # Update state
            current = copy.copy(candidate)
            sample.append(current)
            accept_ratio.append(i)

    print('Accept ratio:', float(len(accept_ratio)) / N)
    return np.array(sample)


a = 0.5
mu1 = 0
mu2 = 0
sigma = 1

N = 30000
burn_in = 0.2

sample = metropolis(N, mu1, mu2, sigma, a)

plt.scatter(
    sample[int(len(sample) * burn_in):, 0],
    sample[int(len(sample) * burn_in):, 1],
    alpha=0.3,
    s=5,
    edgecolor='None'
)
plt.title('MCMC (Metropolis)')
plt.savefig('metropolis.png')

fig = plt.figure(figsize=(15, 6))

ax = fig.add_subplot(121)
plt.hist(sample[int(N * burn_in):, 0], bins=30)
plt.title('x')

ax = fig.add_subplot(122)
plt.hist(sample[int(N * burn_in):, 1], bins=30)
plt.title('y')
plt.savefig('plot.png')

print('x:', np.mean(sample[int(len(sample) * burn_in):, 0]),
      np.var(sample[int(len(sample) * burn_in):, 0]))
# => x: -0.00252259614386 1.26378688755
print('y:', np.mean(sample[int(len(sample) * burn_in):, 1]),
      np.var(sample[int(len(sample) * burn_in):, 1]))
# => y: -0.0174372516771 1.24832585103
