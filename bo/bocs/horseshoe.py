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
    samples = horseshoe()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_title('Beta(0.5,0.5)')
    axes[0].hist(beta, bins=100)
    axes[1].set_title('Horseshoe Distribution')
    axes[1].hist(samples, bins=100)
    fig.tight_layout()
    fig.savefig('horseshoe.png')
    plt.close()

if __name__ == "__main__":
    main()
