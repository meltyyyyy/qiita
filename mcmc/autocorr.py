import numpy as np


def autocorrelation(z, k):
    z_avg = np.mean(z)

    sum_of_covariance = 0
    for i in range(k + 1, len(z)):
        covariance = (z[i] - z_avg) * (z[i - (k + 1)] - z_avg)
        sum_of_covariance += covariance

    sum_of_denominator = 0
    for u in range(len(z)):
        denominator = (z[u] - z_avg)**2
        sum_of_denominator += denominator

    return sum_of_covariance / sum_of_denominator
