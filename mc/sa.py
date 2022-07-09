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
temp = np.array([])
T = 1.
def cool(T): return .90 * T


temp = np.append(temp, T)
for i in range(n_iter):
    T = cool(T)
    temp = np.append(temp, T)

plt.figure()
plt.title('Cooling schedule', fontsize=14)
plt.plot(temp, label='temprature')
plt.xlabel('Iteration')
plt.ylabel('Temprature')
plt.savefig('temprature.png')
plt.close()


# metropolise criterion on each temprature
n_iter = 100
T = 1.
def cool(T): return .99 * T


plt.figure()
plt.title('Criteria', fontsize=14)
for diff in [0.3, 0.1, 0.3, 1.0, 3.0]:
    metro = np.array([])
    for i in range(n_iter):
        T = cool(T)
        criteion = np.exp(diff / T)
        metro = np.append(metro, criteion)
    plt.plot(metro, label=f'{diff}')
plt.xlabel('Iteration')
plt.ylabel('Metropolise criterion')
plt.legend()
plt.savefig('metropolise.png')
plt.close()
