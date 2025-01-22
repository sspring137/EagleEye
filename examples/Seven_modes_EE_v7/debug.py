#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:16:08 2025

@author: sspringe
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

# Directory setup for custom modules
import sys
sys.path.append('../../eagleeye')
import EagleEye_v7

#%%
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
    'grid.color': 'gray'
})

def generate_positions_with_separation(dim, num_anomalies, separation_factor=15):
    positions = []
    center_positions = []
    border_positions = []
    
    for i in range(num_anomalies):
        position = np.zeros(dim)
        
        if i < dim:  # Place the first `dim` anomalies near the center
            position[:dim] = (i - (dim / 2)) * separation_factor
            center_positions.append(position)
        else:  # Place remaining anomalies near the borders
            axis = (i % dim)  # Alternate through dimensions for placement
            direction = (-1 if (i // dim) % 2 == 0 else 1)  # Alternate directions
            position[axis] = direction * (50 + ((i // dim) * separation_factor))
            border_positions.append(position)
        
        positions.append(position)

    return np.array(positions)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generate data with anomalies
num_points = 30000
anomaly_sizes_o = [50, 100, 200, 300, 500, 700, 900]
anomaly_sizes_u = [ 100, 300, 700, ]
n_dim = 3


# Generate the points
reference_data = np.random.uniform(low=-100, high=100, size=(num_points-sum(anomaly_sizes_u), n_dim))

# Generate the points
data_with_anomaly = np.random.uniform(low=-100, high=100, size=(num_points-sum(anomaly_sizes_o), n_dim))

# Generate separated positions dynamically
positions = generate_positions_with_separation(n_dim, len(anomaly_sizes_o)+len(anomaly_sizes_u))

# Generate anomalies with the new positions
overdensities = []
local_i = 1
for size, center in zip(anomaly_sizes_o, positions[1:-2]):
    
    anomaly = np.random.normal(loc=center, scale=local_i, size=(size, n_dim))
    overdensities.append(anomaly)
    local_i = local_i + 1

underdensities = []
local_i = 3
for size, center in zip(anomaly_sizes_u, positions[[0,-2,-1]]):
    
    anomaly = np.random.normal(loc=center, scale=local_i, size=(size, n_dim))
    underdensities.append(anomaly)
    local_i = local_i -1 

# Combine the test set with the anomalies
data_with_anomaly = np.vstack([data_with_anomaly] + overdensities)

# Combine the test set with the anomalies
reference_data = np.vstack([reference_data] + underdensities)

K_M = 400
#%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EagleEye
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Can now split run and post-processing into two steps with critical_qauntiles=None
#importlib.reload(EagleEye_v7)
VALIDATION                                     = reference_data.shape[0]
res = EagleEye_v7.Soar(reference_data, data_with_anomaly, result_dict_in={}, K_M=K_M, critical_quantiles=None, num_cores=100, validation=VALIDATION, partition_size=100,smoothing=3)

#%%

# Just for debugging the post processing
# Reimport the modules
#importlib.reload(EagleEye_v7)

res_new = EagleEye_v7.Soar(reference_data, data_with_anomaly, result_dict_in=res, K_M=K_M, critical_quantiles=[1-1E-4,1-1E-5],  num_cores=100, validation=VALIDATION, partition_size=100,smoothing=3)


#%%


from plot_utilities import (
    plot_injected_anomalies_in_uniform_background,
    plot_data_points_marked_as_putative_anomalies,
    plot_points_extracted_by_iterative_equalization
)

# call them with your data
plot_injected_anomalies_in_uniform_background(
    reference_data, anomaly_sizes_u, data_with_anomaly, anomaly_sizes_o
)

plot_data_points_marked_as_putative_anomalies(
    data_with_anomaly, reference_data, res_new
)

plot_points_extracted_by_iterative_equalization(
    data_with_anomaly, reference_data, res_new, threshold=0.9999
)


#%% # Clustering
clusters = EagleEye_v7.partitian_function(reference_data,data_with_anomaly,res_new,res_new['Upsilon_star_plus'][1], res_new['Upsilon_star_minus'][1],K_M=K_M)

clusters_plus,clusters_minus = clusters

#%%
IV_IE_dict = EagleEye_v7.IV_IE_get_dict(clusters,res_new,[1-1E-5],data_with_anomaly,reference_data)




#%%
from plot_utilities import (
    plot_3d_ie_extra,
    plot_3d_second,
    plot_3d_third
)
# Removed by IE:
plot_3d_ie_extra(IV_IE_dict, data_with_anomaly, reference_data)


# Clustering of anomalous points:
plot_3d_second(IV_IE_dict, data_with_anomaly, reference_data)

# Background from the opposite dataset wrt the clustered anomalies:
plot_3d_third(IV_IE_dict, data_with_anomaly, reference_data)






