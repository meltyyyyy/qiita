import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


a = 0.5
"""
Target distributaion is 2D Gaussian Distribution,
where mu is 0, Sigma^-1 is [1, -a][-a, 1]

P(z1, z2) is target distribution without regulization term.
"""
def P(z1, z2):
    return np.exp(-1 / 2 * (z1**2 - 2 * a * z1 * z2 + z2**2))


xs = []
ys = []
zs = []


def range_ex(start, end, step):
    while start + step < end:
        yield start
        start += step


for i in range_ex(-3, 3, 0.1):
    for j in range_ex(-3, 3, 0.1):
        xs.append(i)
        ys.append(j)
        zs.append(P(i, j))

ax = Axes3D(plt.figure())
ax.scatter3D(xs, ys, zs, s=3, edgecolor='None')
plt.savefig('gaussian_dist_3d.png')
