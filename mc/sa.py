# -*- coding: utf-8 -*-
"""Simulated Annealing
This is Simulated Annealing implementation.
Simulated Annealing (SA) is a probabilistic technique
for approximating the global optimum of a given function.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def objective(x):
    return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x)


n = 10000
data_x = np.linspace(0, 4 * np.pi, n)
data_y = objective(data_x)

# plot objective
plt.figure(figsize=(16, 8))
plt.title('Objective', fontsize=20)
plt.plot(data_x, data_y, label='objective')
plt.savefig('objective.png')
plt.close()


# plot temprature
n_iter = 100
rates = [0.90, 0.95, 0.99]
fig, axes = plt.subplots(nrows=1, ncols=3, tight_layout=True, **{"figsize": (24, 8)})
fig.suptitle('Cooling schedule', fontsize=18)
for i, rate in enumerate(rates):
    temp = np.array([])
    T = 1.
    def cool(T): return rate * T
    for j in range(n_iter):
        T = cool(T)
        temp = np.append(temp, T)
    axes[i].plot(temp, label=f'rate: {rate}')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Temprature')
    axes[i].set_xlim([-5, 105])
    axes[i].set_ylim([-0.1, 1.1])

fig.savefig('temprature.png')


# metropolise criterion on each temprature
n_iter = 100
T = 1.
def cool(T): return .99 * T


plt.figure()
plt.title('Criteria', fontsize=14)
for diff in [0.03, 0.1, 0.3, 1.0, 3.0]:
    metro = np.array([])
    for i in range(n_iter):
        T = cool(T)
        criteion = np.exp(- diff / T)
        metro = np.append(metro, criteion)
    plt.plot(metro, label=f'{diff}')
plt.xlabel('Iteration')
plt.ylabel('Metropolise criterion')
plt.legend()
plt.savefig('metropolise.png')
plt.close()
