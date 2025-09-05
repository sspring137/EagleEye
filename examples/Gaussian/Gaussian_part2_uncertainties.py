#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:09:57 2025

@author: sspringe
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Directory setup for custom modules
import sys
module_path = '../../eagleeye'
sys.path.append(module_path)
import EagleEye
from utils_EE import compute_the_null, partitioning_function

import pickle

# Custom plotting settings
sns.set(style="darkgrid")
plt.rcParams.update({
    'axes.titlesize': 21,
    'axes.labelsize': 17,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 17,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.color': 'silver'
})

# Functions for generating random points and data with anomalies
def generate_random_points(num_points, num_dimensions, radius, shift_factor=.5):
    theta, phi = np.random.uniform(0, 2 * np.pi, (2, num_points))
    x = (radius + radius / 6 * np.cos(phi)) * np.cos(theta) + shift_factor
    y = (radius + radius / 6 * np.cos(phi)) * np.sin(theta)
    z = (radius / 6) * np.sin(phi)
    if num_dimensions > 3:
        mean = np.random.normal(0, radius, num_dimensions - 3)
        covariance = np.eye(num_dimensions - 3) * radius**2
        noise = np.random.multivariate_normal(mean, covariance, num_points)
        points = np.column_stack((x, y, z, noise))
    else:
        points = np.column_stack((x, y))
    return points


def generate_data_with_torus_anomalies(num_dimensions, cluster_sizes, anomaly_radius, shift_factors):
    samples = []
    samples.append(np.random.multivariate_normal(np.zeros(num_dimensions), np.eye(num_dimensions), cluster_sizes[0]))
    samples.append(generate_random_points(cluster_sizes[1], num_dimensions, anomaly_radius, shift_factors))
    if len(cluster_sizes) > 2:
        samples.append(generate_random_points(cluster_sizes[2], num_dimensions, anomaly_radius, shift_factors))
    return np.vstack(samples)


def generate_gaussian_mixture(dim, sizes, means, covariances):
    samples = []
    for mean, cov, size in zip(means, covariances, sizes):
        samples.append(np.random.multivariate_normal(mean, cov, size))
    return np.vstack(samples)


def setup_gaussian_components(num_dimensions=10, background_size=10000, shift_factors=0.5, contamination_size=200):
    m1D1, m1D2 = -shift_factors, 0
    m2D1, m2D2 = +shift_factors, 0
    sizes = [background_size, contamination_size, 1]
    means = [np.zeros(num_dimensions),
             np.array([m1D1, m1D2] + [0] * (num_dimensions - 2)),
             np.array([m2D1, m2D2] + [0] * (num_dimensions - 2))]
    cstd2 = 0.01 + np.random.rand() * 0.02
    return num_dimensions, sizes, means, cstd2

# EagleEye hyperparameters
p = .5
K_M = 500
p_ext = 1e-5
n_jobs = 10

# Precompute null stats
stats_null = compute_the_null(p=p, K_M=K_M)

# Settings
contamination_sizes = [1000, 750, 500, 250, 150, 70]
num_dimensions = 10
seeds = np.arange(10)  # 10 random seeds

def run_experiment(total_samples):
    repech_counts = {cs: [] for cs in contamination_sizes}
    for seed in seeds:
        np.random.seed(seed)
        for contamination_size in contamination_sizes:
            background_size = total_samples - contamination_size - 1
            sig = 1.0
            sigma_a = .3
            dim, sizes, means, cstd2 = setup_gaussian_components(
                num_dimensions=num_dimensions,
                background_size=background_size,
                shift_factors=sig,
                contamination_size=contamination_size)
            covariances = [np.eye(dim), sigma_a**2 * np.eye(dim), cstd2 * np.eye(dim)]

            X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), sum(sizes))
            # Torus anomalies
            test_data_T = generate_data_with_torus_anomalies(
                num_dimensions=dim, cluster_sizes=sizes,
                anomaly_radius=sigma_a, shift_factors=sig)

            # Soar + Repechage (Torus)
            result_dict_T, _ = EagleEye.Soar(
                X, test_data_T, K_M=K_M, p_ext=p_ext,
                n_jobs=n_jobs, stats_null=stats_null,
                result_dict_in={})
            clusters_T = partitioning_function(
                X, test_data_T, result_dict_T, p_ext=p_ext, Z=2.65)
            EE_book_T = EagleEye.Repechage(
                X, test_data_T, result_dict_T, clusters_T, p_ext=1e-5)
            # robust count handling
            count_T = 0
            for cluster_info in EE_book_T.get('Y_OVER_clusters', []):
                if isinstance(cluster_info, dict):
                    rep = cluster_info.get('Repechaged', None)
                else:
                    # assume integer count
                    rep = cluster_info
                if rep is None:
                    continue
                if isinstance(rep, int):
                    count_T += rep
                else:
                    count_T += len(rep)

            # Gaussian anomalies
            test_data_G = generate_gaussian_mixture(dim, sizes, means, covariances)
            result_dict_G, _ = EagleEye.Soar(
                X, test_data_G, K_M=K_M, p_ext=p_ext,
                n_jobs=n_jobs, stats_null=stats_null,
                result_dict_in={})
            clusters_G = partitioning_function(
                X, test_data_G, result_dict_G, p_ext=p_ext, Z=2.65)
            EE_book_G = EagleEye.Repechage(
                X, test_data_G, result_dict_G, clusters_G, p_ext=1e-5)
            count_G = 0
            for cluster_info in EE_book_G.get('Y_OVER_clusters', []):
                if isinstance(cluster_info, dict):
                    rep = cluster_info.get('Repechaged', None)
                else:
                    rep = cluster_info
                if rep is None:
                    continue
                if isinstance(rep, int):
                    count_G += rep
                else:
                    count_G += len(rep)

            repech_counts[contamination_size].append((count_T, count_G))
    return repech_counts

# Run experiments
repech_10k = run_experiment(10000)
repech_100k = run_experiment(100000)

# Aggregate
cont = np.array(contamination_sizes)

def aggregate(counts):
    mean_T = np.array([np.mean([c[0] for c in counts[csize]]) for csize in contamination_sizes])
    stderr_T = np.array([np.std([c[0] for c in counts[csize]], ddof=1) for csize in contamination_sizes])
    mean_G = np.array([np.mean([c[1] for c in counts[csize]]) for csize in contamination_sizes])
    stderr_G = np.array([np.std([c[1] for c in counts[csize]], ddof=1) for csize in contamination_sizes])
    return mean_T, stderr_T, mean_G, stderr_G

mean_T_10k, stderr_T_10k, mean_G_10k, stderr_G_10k = aggregate(repech_10k)
mean_T_100k, stderr_T_100k, mean_G_100k, stderr_G_100k = aggregate(repech_100k)

# Plot with error bars
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
# 10k
axs[0].errorbar(cont, mean_T_10k, yerr=stderr_T_10k, fmt='-o', label='Torus (10k)')
axs[0].errorbar(cont, mean_G_10k, yerr=stderr_G_10k, fmt='-s', label='Gaussian (10k)')
axs[0].plot(cont, cont, '--', color='k')
axs[0].set_title('Repechage Counts with Uncertainty (10k)')
axs[0].set_xlabel('n_anomaly')
axs[0].set_ylabel('Mean n_repechage')
axs[0].legend()

# 100k
axs[1].errorbar(cont, mean_T_100k, yerr=stderr_T_100k, fmt='-o', label='Torus (100k)')
axs[1].errorbar(cont, mean_G_100k, yerr=stderr_G_100k, fmt='-s', label='Gaussian (100k)')
axs[1].plot(cont, cont, '--', color='k')
axs[1].set_title('Repechage Counts with Uncertainty (100k)')
axs[1].set_xlabel('n_anomaly')
axs[1].legend()

plt.tight_layout()
plt.savefig('repechage_with_uncertainty.pdf')
plt.show()
