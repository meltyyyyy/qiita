import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
np.random.seed(42)


class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        y = np.dot(self.x, self.W) + self.b
        return y

    def backward(self, dy):
        dx = np.dot(dy, self.W.T)
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        return dx

    def update(self, learning_rate=0.01):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class Tanh:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def backward(self, dy):
        dx = dy * (1 - self.y ** 2)
        return dx


class MeanSquaredError:
    def __init__(self):
        self.x = None
        self.t = None

    def forward(self, x, t):
        self.x = x
        self.t = t

        diff = x - t
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, dy):
        x, t = self.x, self.t
        diff = x - t
        dx = dy * diff * (2. / len(diff))
        dt = -dx
        return dx, dt


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, output_size)
        self.tanh1 = Tanh()
        self.tanh2 = Tanh()
        self.mse = MeanSquaredError()

    def forward(self, x):
        x = self.tanh1.forward(self.fc1.forward(x))
        x = self.tanh2.forward(self.fc2.forward(x))
        x = self.fc3.forward(x)
        return x

    def loss(self, x, t):
        return self.mse.forward(x, t)

    def backward(self):
        # backward
        dy = 1
        dx, _ = self.mse.backward(dy)
        dx = self.fc3.backward(dx)
        dx = self.tanh2.backward(dx)
        dx = self.fc2.backward(dx)
        dx = self.tanh1.backward(dx)
        dx = self.fc1.backward(dx)

        return dx

    def update(self):
        self.fc3.update()
        self.fc2.update()
        self.fc1.update()


def objective(x):
    return 10 * np.sin(3 * x) * np.exp(- x ** 2)


if __name__ == "__main__":
    N = 50
    x_data = np.random.uniform(low=-2, high=2, size=N)
    y_data = objective(x_data) + np.random.normal(0, 1, size=N)

    x_lins = np.linspace(-2, 2, 1000)
    y_lins = objective(x_lins)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_data, y_data, 'o', markersize=2, label='observation')
    ax.plot(x_lins, y_lins, label='objective')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend()
    plt.tight_layout()
    plt.savefig('data.png')
    plt.close()
