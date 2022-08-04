# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GPy.models import GPRegression
from GPy.kern import RBF
from GPy.core.gp import GP
from GPy.likelihoods import Poisson
from GPy.inference.latent_function_inference import Laplace
from utils.train_test_split import train_test_split

plt.style.use('seaborn-pastel')

np.random.seed(42)


def objective(x):
    y1_ = np.array(x < -5, dtype=np.int8)
    y2_ = np.array(x > 5, dtype=np.int8)
    return y1_ + y2_


def plot_gpr(model, data, title):
    f_samples = model.posterior_samples_f(data['X'][:, None], size=10).reshape(-1, 10)
    plt.figure(figsize=(16, 8))
    plt.title('Bernoulli', fontsize=20)
    plt.plot(data['X'].values, f_samples)
    plt.savefig(f'{title}')


if __name__ == "__main__":
    data = pd.read_csv('http://kasugano.sakura.ne.jp/images/2016/20161112/data-kubo11a.txt')
    # x_train, x_test, y_train, y_test = train_test_split(data['X'].values, data['Y'].values, test_size=0.3)

    kernel = RBF(input_dim=1, ARD=True)
    gpr = GPRegression(np.linspace(1, 50, 50)[:, None], data['Y'][:, None], kernel)
    gpr.optimize()
    gpr.plot()
    plt.savefig("gpr.png")

    poisson = Poisson()
    laplace = Laplace()
    pgpr = GP(np.linspace(1, 50, 50)[:, None], data['Y'][:, None],
              kernel=kernel, likelihood=poisson, inference_method=laplace)
    pgpr.optimize()
    pgpr.plot()
    plt.savefig("poisson.png")
