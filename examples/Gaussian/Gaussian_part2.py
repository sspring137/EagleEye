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
    cstd1 = sigma_a
    covariances = [np.eye(dim), sigma_a**2 * np.eye(dim), cstd2 * np.eye(dim)]

    X = np.random.multivariate_normal(np.array([0] + [0] * (dim - 1)), np.eye(dim), np.array(sizes).sum())

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
with open("results_10k_unimode.pkl", "wb") as file:
    pickle.dump(results_10k, file)

print("Result saved to results_10k_unimode.pkl")


#%%

num_dimensions = 10
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
    cstd1 = sigma_a
    covariances = [np.eye(dim), sigma_a**2 * np.eye(dim), cstd2 * np.eye(dim)]

    X = np.random.multivariate_normal(np.array([0] + [0] * (dim - 1)), np.eye(dim), np.array(sizes).sum())


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
plt.ylim([-30,2030])
plt.legend(['Repechage Torous 100k','Repechage Gaussian 100k'])
plt.xlabel('n_anomaly')
plt.ylabel('n_repechage')




# Save the result using pickle
with open("results_100k_unimode.pkl", "wb") as file:
    pickle.dump(results_100k, file)

print("Result saved to results_100k_unimode.pkl")







#%%
with open("results_10k_unimode.pkl", "rb") as file:
    results_10k_unimode = pickle.load(file)
    
with open("results_100k_unimode.pkl", "rb") as file:
    results_100k_unimode = pickle.load(file)
    
with open("results_10k_flower.pkl", "rb") as file:
    results_10k_flower = pickle.load(file)
    
with open("results_100k_flower.pkl", "rb") as file:
    results_100k_flower = pickle.load(file)



# Assuming results_10k and results_100k are already loaded as per your data
TFRep_10k = [results_10k_unimode['Torous'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
GFRep_10k = [results_10k_unimode['Gaussian'][cluster]['len_Repechaged'] for cluster in contamination_sizes]

TFRep_100k = [results_100k_unimode['Torous'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
GFRep_100k = [results_100k_unimode['Gaussian'][cluster]['len_Repechaged'] for cluster in contamination_sizes]

###############################################################################
# Create a 1x2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for the first panel (10k data)
axs[0].scatter(contamination_sizes, TFRep_10k)
axs[0].scatter(contamination_sizes, GFRep_10k)
axs[0].plot(contamination_sizes, contamination_sizes, c='k', linestyle='--')
axs[0].set_xlim([-30, 1030])
axs[0].set_ylim([-30, 1530])
axs[0].set_title('$Rep\hat{e}chage$')
axs[0].set_xlabel(r'Contamination of $\mathcal{Y}$')
axs[0].set_ylabel('$Rep\hat{e}chage$  cardinality')
axs[0].legend(['Torous: 10k sample', 'Gaussian: 10k sample'])

# Plot for the second panel (100k data)
axs[1].scatter(contamination_sizes, TFRep_100k)
axs[1].scatter(contamination_sizes, GFRep_100k)
axs[1].plot(contamination_sizes, contamination_sizes, c='k',linestyle='--')
axs[1].set_xlim([-30, 1030])
axs[1].set_ylim([-30, 1530])
axs[1].set_title('$Rep\hat{e}chage$')
axs[1].set_xlabel(r'Contamination of $\mathcal{Y}$')
#axs[1].set_ylabel('$Rep\hat{e}chage$  cardinality')
axs[1].legend(['Torous: 100k sample', 'Gaussian: 100k sample'])

# Adjust layout to prevent overlap
plt.tight_layout()

axs[0].text(0.02, 0.95, 'A', transform=axs[0].transAxes,
        fontsize=21, fontweight='bold', va='top', ha='left')
axs[1].text(0.02, 0.95, 'B', transform=axs[1].transAxes,
        fontsize=21, fontweight='bold', va='top', ha='left')
plt.savefig('Supp_increase_cardinality1.pdf', format='pdf')
# Show the plot
plt.show()



# Assuming results_10k and results_100k are already loaded as per your data
TFRep_10k = [results_10k_flower['Torous'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
GFRep_10k = [results_10k_flower['Gaussian'][cluster]['len_Repechaged'] for cluster in contamination_sizes]

TFRep_100k = [results_100k_flower['Torous'][cluster]['len_Repechaged'] for cluster in contamination_sizes]
GFRep_100k = [results_100k_flower['Gaussian'][cluster]['len_Repechaged'] for cluster in contamination_sizes]


# Create a 1x2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for the first panel (10k data)
axs[0].scatter(contamination_sizes, TFRep_10k)
axs[0].scatter(contamination_sizes, GFRep_10k)
axs[0].plot(contamination_sizes, contamination_sizes, c='k', linestyle='--')
axs[0].set_xlim([-30, 1030])
axs[0].set_ylim([-30, 1530])
axs[0].set_title('$Rep\hat{e}chage$')
axs[0].set_xlabel(r'Contamination of $\mathcal{Y}$')
axs[0].set_ylabel('$Rep\hat{e}chage$  cardinality')
axs[0].legend(['Torous: 10k sample', 'Gaussian: 10k sample'])

# Plot for the second panel (100k data)
axs[1].scatter(contamination_sizes, TFRep_100k)
axs[1].scatter(contamination_sizes, GFRep_100k)
axs[1].plot(contamination_sizes, contamination_sizes, c='k',linestyle='--')
axs[1].set_xlim([-30, 1030])
axs[1].set_ylim([-30, 1530])
axs[1].set_title('$Rep\hat{e}chage$')
axs[1].set_xlabel(r'Contamination of $\mathcal{Y}$')
#axs[1].set_ylabel('$Rep\hat{e}chage$  cardinality')
axs[1].legend(['Torous: 100k sample', 'Gaussian: 100k sample'])

# Adjust layout to prevent overlap
plt.tight_layout()

axs[0].text(0.02, 0.95, 'A', transform=axs[0].transAxes,
        fontsize=21, fontweight='bold', va='top', ha='left')
axs[1].text(0.02, 0.95, 'B', transform=axs[1].transAxes,
        fontsize=21, fontweight='bold', va='top', ha='left')
plt.savefig('Supp_increase_cardinality2.pdf', format='pdf')
# Show the plot
plt.show()



