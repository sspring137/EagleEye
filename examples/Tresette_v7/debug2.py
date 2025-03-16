#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:54:27 2025

@author: johan
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Directory setup for custom modules
import sys
module_path = '../../eagleeye'
sys.path.append(module_path)
import EagleEye_v17
from utils_EE_v17 import compute_the_null, partitioning_function

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

from utils_37 import generate_uniform_with_Gaussian_ove_under

# Generate data with anomalies
num_points = 50000
anomaly_sizes_o = [50, 100, 200, 300, 500, 700, 900]
anomaly_sizes_u = [ 100, 300, 700, ]
n_dim = 3

#reference_data, test_data = generate_uniform_with_Gaussian_ove_under(n_dim, num_points, anomaly_sizes_o, anomaly_sizes_u)
X = np.load('reference_data.npy')
Y = np.load('test_data.npy')

#%%

from utils_37 import plot_injected_anomalies_in_uniform_background

# plot of the generated dataset. In silver the background only
# in red the injected overdensities and in blue the injected underdensities
plot_injected_anomalies_in_uniform_background(X, anomaly_sizes_u, Y, anomaly_sizes_o)

#%%

#%% EagleEye hyperparameters

p       = len(Y)/(len(Y)+len(X))

K_M     = 500

p_ext   = 1e-5

n_jobs  = 10

#%%

stats_null                     = compute_the_null(p=p, K_M=K_M)

#%%

#%% Eagle Soar!
# import time
# t = time.time()
result_dict, stats_null = EagleEye_v17.Soar(X, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})
# elapsed17alt = time.time() - t
# print(f'Elapsed time: {elapsed17alt} seconds')


#%%

#%% Cluter the Putative anomalies

clusters = partitioning_function(X,Y,result_dict,p_ext=p_ext,Z=2.65 )

#%% Repêchage

EE_book = EagleEye_v17.Repechage(X,Y,result_dict,clusters,p_ext=1e-5)

#%%

cont = (np.array(anomaly_sizes_o).sum()).astype(int)
fig = plt.figure(1)
plt.scatter(X[:-cont,0], X[:-cont,1], marker='.', s=1, c='silver', alpha=0.3)
for jj in range(3):
    Putative   = EE_book['X_OVER_clusters'][jj]['Putative']
    Pruned     = EE_book['X_OVER_clusters'][jj]['Pruned']
    Repechaged = EE_book['X_OVER_clusters'][jj]['Repechaged']
    Background = EE_book['X_OVER_clusters'][jj]['Background']
    
    # Plotting the scatterplots

    plt.scatter(X[Putative,0], X[Putative,1], marker='.', s=1, c='red', alpha=0.7)
    plt.scatter(X[Repechaged,0], X[Repechaged,1], marker='.', s=1, c='limegreen', alpha=0.7)
    plt.scatter(X[Pruned,0], X[Pruned,1], marker='.', s=1, c='darkgreen', alpha=0.7)

# Displaying the combined plot
plt.show()


#%%

cont = (np.array(anomaly_sizes_o).sum()).astype(int)
fig = plt.figure(2)
plt.scatter(Y[:-cont,0], Y[:-cont,1], marker='.', s=1, c='silver', alpha=0.3)
for jj in range(7):
    Putative   = EE_book['Y_OVER_clusters'][jj]['Putative']
    Pruned     = EE_book['Y_OVER_clusters'][jj]['Pruned']
    Repechaged = EE_book['Y_OVER_clusters'][jj]['Repechaged']
    Background = EE_book['Y_OVER_clusters'][jj]['Background']
    cont = 3000

    # Plotting the scatterplots
    
    plt.scatter(Y[Putative,0], Y[Putative,1], marker='.', s=1, c='red', alpha=0.7)
    plt.scatter(Y[Repechaged,0], Y[Repechaged,1], marker='.', s=1, c='limegreen', alpha=0.7)
    plt.scatter(Y[Pruned,0], Y[Pruned,1], marker='.', s=1, c='darkgreen', alpha=0.7)

# Displaying the combined plot
plt.show()



#%%

# Create a set of banned indices from all clusters
banned_Y = {
    pruned_index
    for cluster in EE_book['Y_OVER_clusters'].values()
    if cluster['Pruned'] is not None  # Check if there's a pruned list
    for pruned_index in cluster['Pruned']
}

#%%

nx = X.shape[0]
# Now create equalized_Y by including indices that are not banned
equalized_Y = [x for x in range(Y.shape[0]) if x not in banned_Y]

equalized_neighbors = result_dict['Knn_model'].kneighbors(Y[equalized_Y, :])[1]
mask = ~np.isin(equalized_neighbors, [x +nx for x in banned_Y])

#%%

filtered_equalized_neighbors = np.empty((equalized_neighbors.shape[0], K_M), dtype=int)

for i in range(equalized_neighbors.shape[0]):
    # Apply the mask for the current row to filter out banned entries
    valid_neighbors = equalized_neighbors[i, :][mask[i, :]]
    # Optionally, handle cases where there are fewer than K_M valid neighbors
    if valid_neighbors.size < K_M:
        raise ValueError(f"Row {i} does not have enough valid neighbors.")
    # Take the first K_M valid neighbors for the current row
    filtered_equalized_neighbors[i, :] = valid_neighbors[:K_M]
    
#%%
    
binary_seq                     = (filtered_equalized_neighbors > X.shape[0]).astype(int)
from EagleEye_v17 import PValueCalculator
KSTAR_RANGE = range(20,K_M)
p_val_info_eq                  = PValueCalculator(binary_seq, KSTAR_RANGE, p=p)

Upsilon_i_equalized_Y = p_val_info_eq.min_pval_plus