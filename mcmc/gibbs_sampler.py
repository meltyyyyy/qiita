import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def range_ex(start, end, step):
    while start + step < end:
        yield start
        start += step


# P(x) : Target distribution
def P(x1, x2, a):
    return np.exp(-1 / 2 * (x1**2 - 2 * a * x1 * x2 + x2**2))


xs = []
ys = []
zs = []
a = 0.5

for i in range_ex(-3, 3, 0.1):
    for j in range_ex(-3, 3, 0.1):
        xs.append(i)
        ys.append(j)
        zs.append(P(i, j, a))

ax = Axes3D(plt.figure())
ax.scatter3D(xs, ys, zs, s=3, edgecolor='None')
plt.savefig('gaussian_dist_3d.png')


def gibbs_sampler(a, step):
    x = 3.5 * np.ones(2)  # initialize
    samples = x
    for i in range(step):
        x[0] = np.random.normal(a * x[1], 1)  # mu=ax[1], sigma=1
        samples = np.append(samples, (x))
        x[1] = np.random.normal(a * x[0], 1)
        samples = np.append(samples, (x))

    samples = samples.reshape((2 * step + 1, x.shape[0]))  # +1 means initial x
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
plt.savefig('gibbs_sampler.png')
