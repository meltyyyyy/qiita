import numpy as np
from core import Model
import layers as L
import functions as F


class BNN(Model):
    def __init__(self, sigma_w, sigma_y):
        super().__init__()
        self.sigma_w = sigma_w
        self.sigma_y = sigma_y

        self.fc1 = L.Linear(in_size=1, out_size=1 * 32)
        self.fc1.W = np.random.normal(0, sigma_w ** 2, size=32).reshape(1, 32)
        self.fc1.b = np.random.normal(0, sigma_w ** 2, size=32)
        self.fc2 = L.Linear(in_size=32, out_size=32)
        self.fc2.W = np.random.normal(0, sigma_w ** 2, size=32 * 32).reshape(32, 32)
        self.fc2.b = np.random.normal(0, sigma_w ** 2, size=32 * 32)
        self.fc3 = L.Linear(in_size=32, out_size=1)
        self.fc3.W = np.random.normal(0, sigma_w ** 2, size=32 * 1).reshape(32, 1)
        self.fc3.b = np.random.normal(0, sigma_w ** 2, size=32 * 1).reshape(32, 1)
        self.tanh = F.tanh

    def forward(self, X, Y=None):
        X = self.tanh(self.fc1(X))
        X = self.tanh(self.fc2(X))
        mu = self.fc3(X)
        return mu + np.random.normal(0, self.sigma_y)
