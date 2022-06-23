# -*- coding: utf-8 -*-
"""Hamiltonian Monte Carlo methods
This is Hamiltonian Monte Carlo method implementation.
Metropolis-Hastings methods is basic algorithm for Markov chain Monte Carlo (MCMC) methods.
<Reference>
https://fisproject.jp/2015/12/mcmc-in-python/ (accessed 21 Jun 2022).
"""

import numpy as np
import matplotlib.pyplot as plt

from autocorr import autocorrelation


# Potential energy for the object
def U(z):
    return 0.5 * (z[0]**2 - 2 * a * z[0] * z[1] + z[1]**2)


def dU_dz(z):
    return (z[0] - a * z[1], z[0] - a * z[1])


# Kinetic energy for the object
def K(p):
    return 0.5 * (p[0]**2 + p[1]**2)


def hamiltonian(p, z):
    return U(z) + K(p)


def leapfrog_half_p(p, z, eps):
    diff = dU_dz(z)
    return (p[0] - 0.5 * eps * diff[0], p[1] - 0.5 * eps * diff[1])


def leapfrog_z(p, z, eps):
    return (z[0] + eps * p[0], z[1] + eps * p[1])


def hmc_sampler(N, L=100, eps=0.01):
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

    return samples


N = 3000
a = 0.5

samples = np.array(hmc_sampler(N))

# plot scatter
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], s=10, c='pink', alpha=0.2,
            edgecolor='red', label='Samples obtained by HMC method')
plt.plot(samples[0:30, 0], samples[0:30, 1], color='green',
         linestyle='dashed', label='First 30 samples')
plt.scatter(samples[0, 0], samples[0, 1], s=50,
            c='b', marker='*', label='initial value')
plt.legend(loc=4, prop={'size': 10})
plt.title('Hamiltonian Monte Carlo method')
plt.savefig('hmc_sampler.png')

# plot autocorreration graph
acorr_data = []
for i in range(100):
    acorr_data.append(autocorrelation(samples, i))
acorr_data = np.asarray(acorr_data)
plt.figure()
markerline, stemlines, baseline = plt.stem(np.arange(
    100), acorr_data[0:100, 0], linefmt="--", use_line_collection=True)
markerline.set_color("red")
markerline.set_markerfacecolor("none")
markerline.set_markersize(2.5)
stemlines.set_color("pink")
baseline.set_color("orange")
plt.savefig('hmc_autocorr.png')

# plot distribution
fig = plt.figure(figsize=(15, 6))

ax = fig.add_subplot(121)
plt.hist(samples[30:, 0], bins=30)
plt.title('x')

ax = fig.add_subplot(122)
plt.hist(samples[30:, 1], bins=30)
plt.title('y')
plt.savefig('hmc_dist.png')
