# -*- coding: utf-8 -*-
"""Simulated Annealing
This is Simulated Annealing implementation.
Simulated Annealing (SA) is a probabilistic technique
for approximating the global optimum of a given function.
"""

from PIL import Image
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
# plt.text(3.6, 7.2, 'global optima', fontdict={'fontsize': 16, 'fontweight': 'bold'})
# plt.text(4.8, -0.4, 'local optima', fontdict={'fontsize': 16, 'fontweight': 'bold'})
# plt.text(8.0, 3.6, 'local optima', fontdict={'fontsize': 16, 'fontweight': 'bold'})
plt.savefig('objective.png')
plt.close()


# plot temprature
n_iter = 100
rates = [0.90, 0.95, 0.99]
fig, axes = plt.subplots(
    nrows=1, ncols=3, tight_layout=True, **{"figsize": (24, 8)})
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
    axes[i].legend()
fig.savefig('temprature.png')


# metropolise criterion on each temprature
n_iter = 100
rates = [0.90, 0.95, 0.99]

fig, axes = plt.subplots(
    nrows=1, ncols=3, tight_layout=True, **{"figsize": (24, 8)})
fig.suptitle('Metropolise Criterion', fontsize=18)

for i, rate in enumerate(rates):
    T = 1.
    def cool(T): return rate * T

    for diff in [0.03, 0.1, 0.3, 1.0, 3.0]:
        metro = np.array([])

        for _ in range(n_iter):
            T = cool(T)
            criteion = np.exp(- diff / T)
            metro = np.append(metro, criteion)
        axes[i].plot(metro, label=f'diff: {diff}')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Metropolise criterion')
    axes[i].legend()

fig.savefig('metropolise.png')


# implementation of simulated annealing
def plot_sa(iter, curr_x, curr_obj, new_x, new_y):
    plt.figure(figsize=(24, 8))
    plt.title('Simulated Annealing', fontsize=20)
    plt.plot(data_x, data_y, label='objective')
    plt.scatter(curr_x, curr_obj, color='blue', label='current x')
    plt.scatter(curr_x, curr_obj, color='green', label='proposed x')
    plt.legend(
        loc='lower left',
        fontsize=12)
    plt.savefig(f'figs/sa_{iter}.png')
    plt.close()


def simulated_annealinng(objective, cooling_rate, n_iter):
    x_iter = np.zeros((n_iter, ))
    obj_iter = np.zeros((n_iter, ))

    # set initial temperature and cooling schedule
    T = 1.
    def cool(T): return cooling_rate * T

    # set initial index to 7000,
    # because this point is around minima
    index = 7000
    curr_x = data_x[index]
    curr_obj = objective(curr_x)

    best_x = curr_x
    best_obj = curr_obj

    for i in range(n_iter):

        # decrease T according to cooling schedule
        T = cool(T)

        index = np.random.choice(n, 1)
        new_x = data_x[index]
        new_obj = objective(new_x)

        # update current solution iterate
        if (new_obj > curr_obj) or (np.random.rand()
                                    < np.exp((new_obj - curr_obj) / T)):
            curr_x = new_x
            curr_obj = new_obj

        # Update best solution
        if new_obj > best_obj:
            best_x = new_x
            best_obj = new_obj

        # save solution
        x_iter[i] = best_x
        obj_iter[i] = best_obj

        plot_sa(i, curr_x, curr_obj, new_x, new_obj)

    return x_iter, obj_iter


x_iter, obj_iter = simulated_annealinng(
    objective, cooling_rate=0.9, n_iter=100)
best_index = np.argmax(obj_iter)
print("best x: {}, best objective: {}".format(
    x_iter[best_index], obj_iter[best_index]))

# create gif
pictures = []

for i in range(n_iter):
    pic_name = 'figs/sa_' + str(i) + '.png'
    img = Image.open(pic_name)
    pictures.append(img)

pictures[0].save('sa.gif', save_all=True, append_images=pictures[1:],
                 optimize=False, duration=500, loop=0)
