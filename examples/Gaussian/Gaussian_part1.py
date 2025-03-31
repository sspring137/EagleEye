#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:52:41 2025

@author: sspringe
"""

# Imports
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

#%%


# Functions for generating random points and data with anomalies
def generate_random_points(num_points, num_dimensions, radius, shift_factor=.5):
    """Generates random points in 3D with specified characteristic scale."""
    theta, phi = np.random.uniform(0, 2 * np.pi, (2, num_points))
    x = (radius + radius / 6 * np.cos(phi)) * np.cos(theta) + shift_factor
    y = (radius + radius / 6 * np.cos(phi)) * np.sin(theta) 
    z = (radius / 6) * np.sin(phi)
    if num_dimensions > 3:
        #noise = np.random.normal(0, radius / 2, (num_points, num_dimensions - 3))
        mean = np.random.normal(0, radius, num_dimensions - 3)
        covariance = np.eye(num_dimensions - 3) *radius**2  
        noise = np.random.multivariate_normal(mean, covariance, num_points)
        
        points = np.column_stack((x, y, z, noise))
    else:
        points = np.column_stack((x, y))
    return points



#def generate_donuts(dim, sizes,R, sig):
def generate_data_with_torus_anomalies(num_dimensions, cluster_sizes, anomaly_radius, shift_factors):
    samples = []
    
    samples.append(np.random.multivariate_normal(np.array([0] + [0] * (num_dimensions - 1)), np.eye(num_dimensions), sizes[0]))
    samples.append( generate_random_points(cluster_sizes[1], num_dimensions,anomaly_radius, shift_factors) )
    if len(sizes)>2:
        samples.append( generate_random_points(cluster_sizes[2], num_dimensions,anomaly_radius, shift_factors) )
    return np.vstack(samples)


def generate_gaussian_mixture(dim, sizes, means, covariances):
    samples = []
    for mean, cov, size in zip(means, covariances, sizes):
        samples.append(np.random.multivariate_normal(mean, cov, size))
    return np.vstack(samples)


#def setup_gaussian_components(dim=10,s1=3000,sig=0.5):
def setup_gaussian_components(num_dimensions=10, background_size=10000, shift_factors = 0.5, contamination_size=200):
    m1D1, m1D2 = -shift_factors, 0
    m2D1, m2D2 = +shift_factors, 0
    sizes = [background_size, contamination_size,1]
    means = [np.array([0] * (num_dimensions)), np.array([m1D1, m1D2] + [0] * (num_dimensions - 2)), np.array([m2D1, m2D2] + [0] * (num_dimensions - 2))]
    cstd2 = 0.01 + np.random.rand() * 0.02
    return num_dimensions, sizes, means, cstd2



import matplotlib.pyplot as plt


def generate_gaussian_mixture_flower(central_points=10000, 
                                               petal_points=15000, 
                                               n_petals=6, 
                                               dim=10, 
                                               radius=5.0, 
                                               random_seed=42):
    """
    Generates a 10-dimensional Gaussian mixture dataset where:
      - A central cluster of `central_points` drawn from N(0, I), ensuring a standard deviation
        of 1 in every dimension.
      - `n_petals` peripheral clusters, each with `petal_points`. Each petal's mean is defined
        using the 2D "flower" pattern for each pair of consecutive dimensions (0-1, 2-3, ..., dim-2, dim-1).
        This produces a flower-like pattern in each dimension pair.
    
    Parameters:
      central_points (int): Number of points in the central cluster (default: 10,000).
      petal_points (int): Number of points in each peripheral cluster (default: 15,000).
      n_petals (int): Number of peripheral clusters (default: 6).
      dim (int): Dimensionality of the data (default: 10).
      radius (float): Radius used to scale the petal means (default: 5.0).
      random_seed (int): Seed for reproducibility (default: 42).
    
    Returns:
      data (np.ndarray): Array of shape ((central_points + n_petals * petal_points), dim) with the data.
      labels (np.ndarray): Array of cluster labels for each data point.
    """
    np.random.seed(random_seed)
    
    data_list = []
    labels_list = []
    
    # Central cluster: drawn from N(0, I)
    central_mean = np.zeros(dim)
    central_cov = np.eye(dim)
    central_data = np.random.multivariate_normal(central_mean, central_cov, central_points)
    data_list.append(central_data)
    labels_list.extend([0] * central_points)
    
    # Parameters to distribute the petal means
    for i in range(n_petals):
        angle = 2 * np.pi * i / n_petals
        # Generate the petal means in pairs of dimensions
        petal_data = np.zeros((petal_points, dim))
        for d in range(0, dim, 2):
            # Each pair (d, d+1) will form one flower pattern
            mean = np.zeros(2)
            mean[0] = radius * np.cos(angle)
            mean[1] = radius * np.sin(angle)
            petal_data[:, d:d+2] = np.random.multivariate_normal(mean, np.eye(2), petal_points)
        
        data_list.append(petal_data)
        labels_list.extend([i + 1] * petal_points)
    
    data = np.vstack(data_list)
    labels = np.array(labels_list)
    
    return data

#%% EagleEye hyperparameters

p       = .5

K_M     = 500

p_ext   = 1e-5

n_jobs  = 10
#%%
stats_null                     = compute_the_null(p=p, K_M=K_M)
#%%

contamination_sizes=[1000, 750, 500, 250, 150, 70]

results_10k = {
"Torous": {
    i: {"len_Pruned": None, "len_Repechaged": None,"Upsilon_star": None}
    for i in contamination_sizes
},
"Gaussian": {
    i: {"len_Pruned": None, "len_Repechaged": None,"Upsilon_star": None}
    for i in contamination_sizes
}
}

#%%
num_dimensions = 10
tot_samples = 10000

#%%



for contamination_size in contamination_sizes:
    background_size=tot_samples-contamination_size-1
    #loop over different center locations
    sig=1.
    sigma_a = .3

    dim, sizes, means, cstd2 = setup_gaussian_components(num_dimensions=num_dimensions, background_size=background_size, shift_factors = sig, contamination_size=contamination_size )
    sizes = [0,contamination_size,0]
    cstd1 = sigma_a
    covariances = [np.eye(dim), sigma_a**2 * np.eye(dim), cstd2 * np.eye(dim)]

    X =  generate_gaussian_mixture_flower(central_points=10000, petal_points=0, n_petals=6, dim=10, radius=5, random_seed=17)
        

    test_data_G = np.concatenate((generate_gaussian_mixture_flower(central_points=10000-contamination_size, petal_points=0, n_petals=6, dim=10, radius=5, random_seed=1), generate_gaussian_mixture(dim, sizes, means, covariances)))
    test_data_T = np.concatenate((generate_gaussian_mixture_flower(central_points=10000-contamination_size, petal_points=0, n_petals=6, dim=10, radius=5, random_seed=2), generate_data_with_torus_anomalies(num_dimensions=dim, cluster_sizes=sizes, anomaly_radius=sigma_a, shift_factors=sig)))
    
    
    result_dict, stats_null = EagleEye.Soar(X, test_data_T, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})

    #%% Cluter the Putative anomalies
    
    clusters = partitioning_function(X,test_data_T,result_dict,p_ext=p_ext,Z=2.65 )
    
    #%% Repêchage
    
    EE_book = EagleEye.Repechage(X,test_data_T,result_dict,clusters,p_ext=1e-5)
    

    result_dict_G, stats_null = EagleEye.Soar(X, test_data_G, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})

    #%% Cluter the Putative anomalies
    
    clusters_G = partitioning_function(X,test_data_G,result_dict,p_ext=p_ext,Z=2.65 )
    
    #%% Repêchage
    
    EE_book_G = EagleEye.Repechage(X,test_data_G,result_dict,clusters,p_ext=1e-5)
    

        
    results_10k['Torous'][contamination_size]['len_Repechaged'] = sum(len(EE_book['Y_OVER_clusters'][clust]['Repechaged']) if EE_book['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters'])))
    results_10k['Torous'][contamination_size]['len_Pruned']  = sum(len(EE_book['Y_OVER_clusters'][clust]['Pruned']) if EE_book['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters'])))
    results_10k['Torous'][contamination_size]['Upsilon_star']  = result_dict['Upsilon_star_plus'][result_dict['p_ext']]

    results_10k['Gaussian'][contamination_size]['len_Repechaged'] = sum(len(EE_book_G['Y_OVER_clusters'][clust]['Repechaged']) if EE_book_G['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters'])))
    results_10k['Gaussian'][contamination_size]['len_Pruned'] = sum(len(EE_book_G['Y_OVER_clusters'][clust]['Pruned']) if EE_book_G['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters'])))
    results_10k['Gaussian'][contamination_size]['Upsilon_star'] = result_dict['Upsilon_star_plus'][result_dict['p_ext']]



# Save the result using pickle
with open("results_10k_flower.pkl", "wb") as file:
    pickle.dump(results_10k, file)

print("Result saved to results_10k_flower.pkl")


#%%

# num_dimensions = 10
tot_samples_100k = 100000
#%%

results_100k = {
"Torous": {
    i: {"len_Pruned": None, "len_Repechaged": None,"Upsilon_star": None}
    for i in contamination_sizes
},
"Gaussian": {
    i: {"len_Pruned": None, "len_Repechaged": None,"Upsilon_star": None}
    for i in contamination_sizes
}
}

#%%

for contamination_size in contamination_sizes:
    background_size_100k=tot_samples_100k-contamination_size-1
    #loop over different center locations
    sig=1.
    sigma_a = .3

    dim, sizes, means, cstd2 = setup_gaussian_components(num_dimensions=num_dimensions, background_size=background_size_100k, shift_factors = sig, contamination_size=contamination_size )
    sizes = [0,contamination_size,0]
    cstd1 = sigma_a
    covariances = [np.eye(dim), sigma_a**2 * np.eye(dim), cstd2 * np.eye(dim)]

    X =  generate_gaussian_mixture_flower(central_points=10000, petal_points=15000, n_petals=6, dim=10, radius=5, random_seed=17)
        

    test_data_G = np.concatenate((generate_gaussian_mixture_flower(central_points=10000-contamination_size, petal_points=15000, n_petals=6, dim=10, radius=5, random_seed=11), generate_gaussian_mixture(dim, sizes, means, covariances)))
    test_data_T = np.concatenate((generate_gaussian_mixture_flower(central_points=10000-contamination_size, petal_points=15000, n_petals=6, dim=10, radius=5, random_seed=12), generate_data_with_torus_anomalies(num_dimensions=dim, cluster_sizes=sizes, anomaly_radius=sigma_a, shift_factors=sig)))
    
    

    result_dict, stats_null = EagleEye.Soar(X, test_data_T, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})

    #%% Cluter the Putative anomalies
    
    clusters = partitioning_function(X,test_data_T,result_dict,p_ext=p_ext,Z=2.65 )
    
    #%% Repêchage
    
    EE_book = EagleEye.Repechage(X,test_data_T,result_dict,clusters,p_ext=1e-5)
    

    result_dict_G, stats_null = EagleEye.Soar(X, test_data_G, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})

    #%% Cluter the Putative anomalies
    
    clusters_G = partitioning_function(X,test_data_G,result_dict,p_ext=p_ext,Z=2.65 )
    
    #%% Repêchage
    
    EE_book_G = EagleEye.Repechage(X,test_data_G,result_dict,clusters,p_ext=1e-5)
    
        
    results_100k['Torous'][contamination_size]['len_Repechaged'] = sum(len(EE_book['Y_OVER_clusters'][clust]['Repechaged']) if EE_book['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters'])))
    results_100k['Torous'][contamination_size]['len_Pruned']  = sum(len(EE_book['Y_OVER_clusters'][clust]['Pruned']) if EE_book['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters'])))
    results_100k['Torous'][contamination_size]['Upsilon_star']  = result_dict['Upsilon_star_plus'][result_dict['p_ext']]

    results_100k['Gaussian'][contamination_size]['len_Repechaged'] = sum(len(EE_book_G['Y_OVER_clusters'][clust]['Repechaged']) if EE_book_G['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters'])))
    results_100k['Gaussian'][contamination_size]['len_Pruned'] = sum(len(EE_book_G['Y_OVER_clusters'][clust]['Pruned']) if EE_book_G['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters'])))
    results_100k['Gaussian'][contamination_size]['Upsilon_star'] = result_dict['Upsilon_star_plus'][result_dict['p_ext']]

#%%

TFRep_10k = [results_10k['Torous'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
GFRep_10k = [results_10k['Gaussian'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
TFPru_10k = [results_10k['Torous'][cluster]['len_Pruned'] for cluster in contamination_sizes]
GFPru_10k = [results_10k['Gaussian'][cluster]['len_Pruned'] for cluster in contamination_sizes]

plt.figure()
plt.scatter(contamination_sizes, TFPru_10k)
plt.scatter(contamination_sizes, GFPru_10k)
plt.plot(contamination_sizes,contamination_sizes, c='k')
plt.xlim([-30,1030])
plt.ylim([-30,1030])
plt.legend(['Pruned Torous 10k','Pruned Gaussian 10k'])
plt.xlabel('n_anomaly')
plt.ylabel('n_equalized')

plt.figure()
plt.scatter(contamination_sizes, TFRep_10k)
plt.scatter(contamination_sizes, GFRep_10k)
plt.plot(contamination_sizes,contamination_sizes, c='k')
plt.xlim([-30,1030])
plt.ylim([-30,1030])
plt.legend(['Repechage Torous 10k','Repechage Gaussian 10k'])
plt.xlabel('n_anomaly')
plt.ylabel('n_repechage')
#%%

TFRep_100k = [results_100k['Torous'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
GFRep_100k = [results_100k['Gaussian'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
TFPru_100k = [results_100k['Torous'][cluster]['len_Pruned'] for cluster in contamination_sizes]
GFPru_100k = [results_100k['Gaussian'][cluster]['len_Pruned'] for cluster in contamination_sizes]

plt.figure()
plt.scatter(contamination_sizes, TFPru_100k)
plt.scatter(contamination_sizes, GFPru_100k)
plt.plot(contamination_sizes,contamination_sizes, c='k')
plt.xlim([-30,1030])
plt.ylim([-30,1030])
plt.legend(['Pruned Torous 100k','Pruned Gaussian 100k'])
plt.xlabel('n_anomaly')
plt.ylabel('n_equalized')

plt.figure()
plt.scatter(contamination_sizes, TFRep_100k)
plt.scatter(contamination_sizes, GFRep_100k)
plt.plot(contamination_sizes,contamination_sizes, c='k')
plt.xlim([-30,1030])
plt.ylim([-30,1030])
plt.legend(['Repechage Torous 100k','Repechage Gaussian 100k'])
plt.xlabel('n_anomaly')
plt.ylabel('n_repechage')




# Save the result using pickle
with open("results_100k_flower.pkl", "wb") as file:
    pickle.dump(results_100k, file)

print("Result saved to results_100k_flower.pkl")








