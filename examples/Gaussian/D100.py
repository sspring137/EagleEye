#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:59:33 2025

@author: sspringe
"""
import numpy as np
import matplotlib.pyplot as plt



def generate_data(n_points, contamination_percentage, random_state=0):
    """
    Generate synthetic data for anomaly detection with transformed features,
    then pad to 100 dims and apply a unitary rotation.

    Returns:
    - data: ndarray of shape (n_points, 10), original generated features.
    - transformed: ndarray of shape (n_points, 100), rotated features.
    - labels: ndarray of shape (n_points,), 0 for normal points, 1 for anomalies.
    """
    rng = np.random.RandomState(random_state)
    # Determine counts
    normal_count = int(np.floor(n_points * (100 - contamination_percentage) / 100.0))
    anomaly_count = n_points - normal_count

    # Generate normal points: uniform in [-10, 10]
    normal_data = rng.uniform(-10, 10, size=(normal_count, 10))

    # Define a fixed covariance matrix with some correlations for anomalies
    cov = np.eye(10)
    cov[0, 1] = cov[1, 0] = 0.8
    cov[2, 3] = cov[3, 2] = 0.9
    cov[7, 5] = cov[5, 7] = -0.7

    # Generate anomalies: Gaussian at mean=3 with that covariance
    anomalies = rng.multivariate_normal(mean=np.ones(10), cov=cov, size=anomaly_count)
    anomalies = anomalies
    # Combine and label
    data = np.vstack([normal_data, anomalies])
    labels = np.hstack([np.zeros(normal_count), np.ones(anomaly_count)])

    # # Shuffle
    # idx = rng.permutation(n_points)
    # data = data[idx]
    # labels = labels[idx]

    # --- original 10 nonlinear transforms ---
    transformed = np.zeros_like(data)
    W = rng.randn(10)
    b = 2 * np.pi * rng.rand()

    # 1) Scaled arctan dims 0+1
    transformed[:, 0] = (2/np.pi) * np.arctan(data[:, 0] + data[:, 1])
    # 2) Tanh of half dim2
    transformed[:, 1] = np.tanh(0.5 * data[:, 2])
    # 3) Centered sigmoid of dim3–dim4
    sig = 1.0 / (1.0 + np.exp(-(data[:, 3] - data[:, 4])))
    transformed[:, 2] = 2 * sig - 1
    # 4) Rational x/(1+|x|) on dim5
    transformed[:, 3] = data[:, 5] / (1 + np.abs(data[:, 5]))
    # 5) RBF bump on dim6–dim7
    diff = data[:, 6] - data[:, 7]
    transformed[:, 4] = np.exp(-(diff**2) / 2)
    # 6) Random Fourier feature
    transformed[:, 5] = np.cos(data.dot(W) + b)
    # 7) Cross-term ratio on dims 8×9
    prod89 = data[:, 8] * data[:, 9]
    transformed[:, 6] = prod89 / (1 + np.abs(prod89))
    # 8) Norm of dims 0&1
    norm01 = np.sqrt(data[:, 0]**2 + data[:, 1]**2)
    transformed[:, 7] = norm01 / (1 + norm01)
    # 9) Log1p of |dim2|
    l2 = np.log1p(np.abs(data[:, 2]))
    transformed[:, 8] = l2 / (1 + l2)
    # 10) Fractional‐power on dim3
    fp3 = np.sign(data[:, 3]) * np.sqrt(np.abs(data[:, 3]))
    transformed[:, 9] = fp3 / (1 + np.abs(fp3))

    # --- pad to 100 dims and rotate ---
    n, d = transformed.shape  # (n_points, 10)
    # 1) pad with 90 zeros → (n_points, 100)
    extended = np.hstack([transformed, np.zeros((n, 90))])

    # 2) build a random orthonormal matrix Q (100×100)
    seed = 111
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((100, 100))
    Q, _ = np.linalg.qr(R)  # Q @ Q.T = I

    # 3) apply the unitary rotation
    transformed = extended.dot(Q)

    return data, transformed, extended, labels



# # Example usage
# n_points = 10000
# contamination_percentage = 1  # 5% anomalies
# data, transformed, extended, labels = generate_data(n_points, contamination_percentage, random_state=42)

# # Quick 3D scatter of transformed dims 0,1,3
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# dimsss=[0,1,2]
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(transformed[labels == 0, dimsss[0]],
#            transformed[labels == 0, dimsss[1]],
#            transformed[labels == 0, dimsss[2]],
#            label='Normal', alpha=0.7)
# ax.scatter(transformed[labels == 1, dimsss[0]],
#            transformed[labels == 1, dimsss[1]],
#            transformed[labels == 1, dimsss[2]],
#            label='Anomaly', alpha=0.7)
# ax.legend()
# plt.show()


# fig = plt.figure()
# dimsss=[0,5,9]
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data[labels == 0, dimsss[0]],
#            data[labels == 0, dimsss[1]],
#            data[labels == 0, dimsss[2]],
#            label='Normal', alpha=0.7)
# ax.scatter(data[labels == 1, dimsss[0]],
#            data[labels == 1, dimsss[1]],
#            data[labels == 1, dimsss[2]],
#            label='Anomaly', alpha=0.7)
# ax.legend()
# plt.show()
