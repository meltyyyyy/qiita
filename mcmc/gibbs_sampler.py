# -*- coding: utf-8 -*-
"""Gibbs Sampling
This is Gibbs Sampling implementation.
Gibbs sampling or a Gibbs sampler is a Markov chain Monte Carlo (MCMC) algorithm
for obtaining a sequence of observations which are approximated
from a specified multivariate probability distribution,
when direct sampling is difficult.
<Reference>
https://research.miidas.jp/2019/12/mcmc%E5%85%A5%E9%96%80-gibbs-sampling/ (accessed 21 Jun 2022).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def gibbs_sampler(a, step):
    z = np.zeros(2)
    samples = z
    for i in range(step):
        z[0] = np.random.normal(a * z[1], 1)
        samples = np.append(samples, (z))
        z[1] = np.random.normal(a * z[0], 1)
        samples = np.append(samples, (z))

    samples = samples.reshape((2 * step + 1, z.shape[0]))
    return samples


step = 3000
a = 0.5
samples = gibbs_sampler(a, step)

plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], s=10, c='pink', alpha=0.2,
            edgecolor='red', label='Samples obtained by Gibbs Sampling')
plt.plot(samples[0:30, 0], samples[0:30, 1], color='green',
         linestyle='dashed', label='First 30 samples')
plt.scatter(samples[0, 0], samples[0, 1], s=50,
            c='b', marker='*', label='initial value')
plt.legend(loc=4, prop={'size': 10})
plt.title('Gibbs sampler')
plt.savefig('gibbs_sampler.png')

# plot distribution
fig = plt.figure(figsize=(15, 6))

ax = fig.add_subplot(121)
plt.hist(samples[30:, 0], bins=30)
plt.title('x')

ax = fig.add_subplot(122)
plt.hist(samples[30:, 1], bins=30)
plt.title('y')
plt.savefig('gibbs_dist.png')
