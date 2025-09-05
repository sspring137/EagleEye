#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:05:15 2025

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


#%% EagleEye hyperparameters

p       = .25

K_M     = 500

p_ext   = 1e-5

n_jobs  = 10

#%%
stats_null                     = compute_the_null(p=p, K_M=K_M)
#%%

contamination_sizes=[1000, 750, 500, 250, 150, 70]

results_10k = {
"Torous": {
    i: {"len_Pruned": [], "len_Repechaged": [],"Upsilon_star": None}
    for i in contamination_sizes
},
"Gaussian": {
    i: {"len_Pruned": [], "len_Repechaged": [],"Upsilon_star": None}
    for i in contamination_sizes
}
}

#%%
num_dimensions = 10
tot_samples = 10000

#%%

for rep in range(10):
    for contamination_size in contamination_sizes:
        background_size=tot_samples-contamination_size-1
        #loop over different center locations
        sig=1.
        sigma_a = .3
    
        dim, sizes, means, cstd2 = setup_gaussian_components(num_dimensions=num_dimensions, background_size=background_size, shift_factors = sig, contamination_size=contamination_size )
        cstd1 = sigma_a
        covariances = [np.eye(dim), sigma_a**2 * np.eye(dim), cstd2 * np.eye(dim)]
    
        X = np.random.multivariate_normal(np.array([0] + [0] * (dim - 1)), np.eye(dim), tot_samples*3)
    
        test_data_G = generate_gaussian_mixture(dim, sizes, means, covariances)
        test_data_T = generate_data_with_torus_anomalies(num_dimensions=dim, cluster_sizes=sizes, anomaly_radius=sigma_a, shift_factors=sig)
    
    
    
        #VALIDATION            = reference_data.shape[0]
    
    
        result_dict, stats_null = EagleEye.Soar(X, test_data_T, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})
    
        #%% Cluter the Putative anomalies
        
        clusters = partitioning_function(X,test_data_T,result_dict,p_ext=p_ext,Z=2.65 )
        
        #%% Repêchage
        
        EE_book = EagleEye.Repechage(X,test_data_T,result_dict,clusters,p_ext=1e-5)
        
    
        result_dict_G, stats_null = EagleEye.Soar(X, test_data_G, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})
    
        #%% Cluter the Putative anomalies
        
        clusters_G = partitioning_function(X,test_data_G,result_dict_G,p_ext=p_ext,Z=2.65 )
        
        #%% Repêchage
        
        EE_book_G = EagleEye.Repechage(X,test_data_G,result_dict_G,clusters,p_ext=1e-5)
        
    
            
        results_10k['Torous'][contamination_size]['len_Repechaged'].append(sum(len(EE_book['Y_OVER_clusters'][clust]['Repechaged']) if EE_book['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters']))) )
        results_10k['Torous'][contamination_size]['len_Pruned'].append(sum(len(EE_book['Y_OVER_clusters'][clust]['Pruned']) if EE_book['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters']))) )
        results_10k['Torous'][contamination_size]['Upsilon_star'] = result_dict['Upsilon_star_plus'][result_dict['p_ext']] 
    
        results_10k['Gaussian'][contamination_size]['len_Repechaged'].append(sum(len(EE_book_G['Y_OVER_clusters'][clust]['Repechaged']) if EE_book_G['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters']))) )
        results_10k['Gaussian'][contamination_size]['len_Pruned'].append(sum(len(EE_book_G['Y_OVER_clusters'][clust]['Pruned']) if EE_book_G['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters']))) )
        results_10k['Gaussian'][contamination_size]['Upsilon_star'] = result_dict_G['Upsilon_star_plus'][result_dict_G['p_ext']] 


# Save the result using pickle
with open("results_10k_p025.pkl", "wb") as file:
    pickle.dump(results_10k, file)

print("Result saved to results_10k_p025.pkl")

#%%





p       = .625

K_M     = 500

p_ext   = 1e-5

n_jobs  = 10

#%%
stats_null                     = compute_the_null(p=p, K_M=K_M)
#%%

contamination_sizes=[1000, 750, 500, 250, 150, 70]

results_10k = {
"Torous": {
    i: {"len_Pruned": [], "len_Repechaged": [],"Upsilon_star": None}
    for i in contamination_sizes
},
"Gaussian": {
    i: {"len_Pruned": [], "len_Repechaged": [],"Upsilon_star": None}
    for i in contamination_sizes
}
}

#%%
num_dimensions = 10
tot_samples = 10000

#%%

for rep in range(10):
    for contamination_size in contamination_sizes:
        background_size=tot_samples-contamination_size-1
        #loop over different center locations
        sig=1.
        sigma_a = .3
    
        dim, sizes, means, cstd2 = setup_gaussian_components(num_dimensions=num_dimensions, background_size=background_size, shift_factors = sig, contamination_size=contamination_size )
        cstd1 = sigma_a
        covariances = [np.eye(dim), sigma_a**2 * np.eye(dim), cstd2 * np.eye(dim)]
    
        X = np.random.multivariate_normal(np.array([0] + [0] * (dim - 1)), np.eye(dim), int(tot_samples*.6))
    
        test_data_G = generate_gaussian_mixture(dim, sizes, means, covariances)
        test_data_T = generate_data_with_torus_anomalies(num_dimensions=dim, cluster_sizes=sizes, anomaly_radius=sigma_a, shift_factors=sig)
    
    
    
        #VALIDATION            = reference_data.shape[0]
    
    
        result_dict, stats_null = EagleEye.Soar(X, test_data_T, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})
    
        #%% Cluter the Putative anomalies
        
        clusters = partitioning_function(X,test_data_T,result_dict,p_ext=p_ext,Z=2.65 )
        
        #%% Repêchage
        
        EE_book = EagleEye.Repechage(X,test_data_T,result_dict,clusters,p_ext=1e-5)
        
    
        result_dict_G, stats_null = EagleEye.Soar(X, test_data_G, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})
    
        #%% Cluter the Putative anomalies
        
        clusters_G = partitioning_function(X,test_data_G,result_dict_G,p_ext=p_ext,Z=2.65 )
        
        #%% Repêchage
        
        EE_book_G = EagleEye.Repechage(X,test_data_G,result_dict_G,clusters,p_ext=1e-5)
        
    
            
        results_10k['Torous'][contamination_size]['len_Repechaged'].append(sum(len(EE_book['Y_OVER_clusters'][clust]['Repechaged']) if EE_book['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters']))) )
        results_10k['Torous'][contamination_size]['len_Pruned'].append(sum(len(EE_book['Y_OVER_clusters'][clust]['Pruned']) if EE_book['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book['Y_OVER_clusters']))) )
        results_10k['Torous'][contamination_size]['Upsilon_star'] = result_dict['Upsilon_star_plus'][result_dict['p_ext']] 
    
        results_10k['Gaussian'][contamination_size]['len_Repechaged'].append(sum(len(EE_book_G['Y_OVER_clusters'][clust]['Repechaged']) if EE_book_G['Y_OVER_clusters'][clust]['Repechaged'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters']))) )
        results_10k['Gaussian'][contamination_size]['len_Pruned'].append(sum(len(EE_book_G['Y_OVER_clusters'][clust]['Pruned']) if EE_book_G['Y_OVER_clusters'][clust]['Pruned'] is not None else 0 for clust in range(len(EE_book_G['Y_OVER_clusters']))) )
        results_10k['Gaussian'][contamination_size]['Upsilon_star'] = result_dict_G['Upsilon_star_plus'][result_dict_G['p_ext']] 


# Save the result using pickle
with open("results_10k_p0625.pkl", "wb") as file:
    pickle.dump(results_10k, file)

print("Result saved to results_10k_p0625.pkl")

