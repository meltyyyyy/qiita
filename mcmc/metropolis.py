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

from autocorr import autocorrelation

a = 0.5
# P(z1, z2) is target distribution without regulization term.


def P(z1, z2):
    return np.exp(-0.5 * (z1**2 - 2 * a * z1 * z2 + z2**2))


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


def metropolis(N, mu1, mu2, sigma):
    z = (0, 0)
    sample = []
    sample.append(z)
    accept = []

    for i in range(N):
        z_new = Q(z, mu1, mu2, sigma)

        T_prev = P(z[0], z[1])
        T_next = P(z_new[0], z_new[1])
        r = T_next / T_prev

        if r > 1 or r > np.random.uniform(0, 1):
            z = copy.copy(z_new)
            sample.append(z)
            accept.append(0)
        else:
            accept.append(1)
    rate = np.mean(accept)
    print(f'acceptance rate : {rate}')
    return np.array(sample)


mu1 = 0
mu2 = 0
sigma = 1

N = 3000

samples = metropolis(N, mu1, mu2, sigma)

# plot scatter
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], s=10, c='pink', alpha=0.2,
            edgecolor='red', label='Samples obtained by M-H method')
plt.plot(samples[0:30, 0], samples[0:30, 1], color='green',
         linestyle='dashed', label='First 30 samples')
plt.scatter(samples[0, 0], samples[0, 1], s=50,
            c='b', marker='*', label='initial value')
plt.legend(loc=4, prop={'size': 10})
plt.title('Metropolis-Hastings method')
plt.savefig('metropolis.png')

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
plt.title('Metropolis-Hastings Autocorrelation')
plt.savefig('metropolis_autocorr.png')

# plot distribution
fig = plt.figure(figsize=(15, 6))

ax = fig.add_subplot(121)
plt.hist(samples[30:, 0], bins=30)
plt.title('x')

ax = fig.add_subplot(122)
plt.hist(samples[30:, 1], bins=30,  color="g")
plt.title('y')
plt.title('Metropolis-Hastings Distribution')
plt.savefig('metropolis_dist.png')
