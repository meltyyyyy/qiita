from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample, DenseNN
import pyro.distributions as dist
import pyro
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

N = 30


def make_data(x, eps):
    y = 10 * np.sin(3 * x) * np.exp(- x ** 2)
    noise = np.random.normal(0, eps, size=x.shape[0])
    return y + noise


x_data = np.random.uniform(low=-2, high=2, size=N)
y_data = make_data(x_data, 2)


x_linspace = np.linspace(-2, 2, 1000)
y_linspace = make_data(x_linspace, 0.0)


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_data, y_data, 'o', markersize=2, label='data')
ax.plot(x_linspace, y_linspace, label='true_func')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.legend()
plt.tight_layout()
plt.savefig('data.png')
plt.close()


h1, h2 = 10, 10


class Model(PyroModule):
    def __init__(self, h1=h1, h2=h2):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](1, h1)
        self.fc1.weight = PyroSample(dist.Normal(
            0., 10.).expand([h1, 1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(
            0., 10.).expand([h1]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(
            0., 10.).expand([h2, h1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(
            0., 10.).expand([h2]).to_event(1))
        self.fc3 = PyroModule[nn.Linear](h2, 1)
        self.fc3.weight = PyroSample(dist.Normal(
            0., 10.).expand([1, h2]).to_event(2))
        self.fc3.bias = PyroSample(
            dist.Normal(0., 10.).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, X, Y=None, h1=h1, h2=h2):
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        mu = self.fc3(X)
        sigma = pyro.sample("sigma", dist.Uniform(0., 2.0))
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("Y", dist.Normal(mu, sigma).to_event(1), obs=Y)
        return mu

# インスタンス化
model = Model(h1=h1, h2=h2)

# パラメータをリセット
pyro.clear_param_store()
# 近似分布の設定
guide = AutoDiagonalNormal(model)
# optimizerの設定
adam = pyro.optim.Adam({"lr": 0.03})
# SVIクラスのインスタンス化
svi = SVI(model, guide, adam, loss=Trace_ELBO())

# データをtensorに変換
x_data = torch.from_numpy(x_data).float().unsqueeze(-1)
y_data = torch.from_numpy(y_data).float().unsqueeze(-1)

# 最適化
torch.manual_seed(0)
n_epoch = 10000
loss_list = []
for epoch in range(n_epoch):

    # 変分推論の最適化ステップ
    loss = svi.step(x_data, y_data, h1, h2)
    loss_list.append(loss)

# 損失関数の可視化
plt.plot(np.array(loss_list))
plt.xlabel('step')
plt.ylabel('Loss')
plt.tight_layout();
plt.savefig('learning_curve.png')
plt.close()
