import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def horseshoe():
    tau = 1.0
    k = np.random.beta(0.5, 0.5, size=10000)
    λ = np.sqrt(-1 + 1 / k) / tau
    scale = (tau ** 2) * (λ ** 2)
    samples = np.random.normal(0, scale)
    return samples


def main():
    beta = np.random.beta(0.5, 0.5, size=100000)
    fig = plt.figure()
    plt.title("Beta(0.5, 0.5)")
    plt.hist(beta, bins=100)
    fig.savefig("dist-k.png")
    plt.close()

    samples = horseshoe()
    fig = plt.figure()
    plt.title("Horseshoe ditribution")
    plt.hist(samples, bins=100)
    fig.savefig("horseshoe.png")
    plt.close()


if __name__ == "__main__":
    main()
