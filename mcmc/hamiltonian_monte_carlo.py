# -*- coding: utf-8 -*-
"""Hamiltonian Monte Carlo methods
This is Hamiltonian Monte Carlo method implementation.
Metropolis-Hastings methods is basic algorithm for Markov chain Monte Carlo (MCMC) methods.
<Reference>
https://fisproject.jp/2015/12/mcmc-in-python/ (accessed 21 Jun 2022).
"""

import numpy as np
import matplotlib.pyplot as plt

a = 0

# P(x) : Target distribution


def P(x1, x2):
    return np.exp(-0.5 * (x1**2 - 2 * a * x1 * x2 + x2**2))


def U(z):
    return 0.5 * (z[0]**2 - 2 * a * z[0] * z[1] + z[1]**2)


def dU_dz(z):
    return (z[0] - a * z[1], z[0] - a * z[1])


def K(p):
    return 0.5 * (p[0]**2 + p[1]**2)


def hamiltonian(p, z):
    return U(z, a) + K(p)


def leapfrog_half_p(p, z, eps):
    diff = dU_dz(z, a)
    return (p[0] - 0.5 * eps * diff[0], p[1] - 0.5 * eps * diff[1])


def leapfrog_z(p, z, eps):
    return (z[0] + eps * p[0], z[1] + eps * p[1])


def hmc_sampler(N=30000, L=100, eps=0.01):
    samples = []
    z = (0, 0)
    p = (np.random.normal(0, 1), np.random.normal(0, 1))

    prev_H = hamiltonian(p, z)
    samples.append(z)

    for t in range(N):
        z_prev = z
        prev_H = hamiltonian(p, z)

        for i in range(L):
            p = leapfrog_half_p(p, z, eps)
            z = leapfrog_z(p, z, eps)
            p = leapfrog_half_p(p, z, eps)

        H = hamiltonian(p, z)
        r = np.exp(prev_H - H)
        if r > 1:
            samples.append(z)
        elif r > 0 and np.random.uniform(0, 1) < r:
            samples.append(z)
        else:
            z = z_prev

        p = (np.random.normal(0, 1), np.random.normal(0, 1))


samples = np.array(hmc_sampler())


burn_in = 0.2

plt.scatter(
    samples[int(len(samples) * burn_in):, 0],
    samples[int(len(samples) * burn_in):, 1],
    alpha=0.3,
    s=5,
    edgecolor='None'
)
plt.title('MCMC (Metropolis)')
plt.savefig('metropolis.png')

fig = plt.figure(figsize=(15, 6))

ax = fig.add_subplot(121)
plt.hist(samples[int(N * burn_in):, 0], bins=30)
plt.title('x')

ax = fig.add_subplot(122)
plt.hist(samples[int(N * burn_in):, 1], bins=30)
plt.title('y')
plt.savefig('plot.png')

print('x:', np.mean(samples[int(len(samples) * burn_in):, 0]),
      np.var(samples[int(len(samples) * burn_in):, 0]))
# => x: -0.00252259614386 1.26378688755
print('y:', np.mean(samples[int(len(samples) * burn_in):, 1]),
      np.var(samples[int(len(samples) * burn_in):, 1]))
# => y: -0.0174372516771 1.24832585103
