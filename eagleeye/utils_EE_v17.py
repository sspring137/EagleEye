#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Fri 7 16:17:33 2025

@author: sspringe
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from EagleEye_v17 import PValueCalculator
     

    
# ----------------------------------------------------------------------
# 1) Get the null
# ----------------------------------------------------------------------

def compute_the_null(p, K_M):
    """
    Helper to get the null distributions.
    """
    KSTAR_RANGE = range(20, K_M)
    stats_null = {
    p: None,
    1-p: None,
    }
    
    if p==0.5:
       binary_sequences_p         = np.random.binomial(n=1, p=p  , size=(1000000, K_M))
       p_val_info                 = PValueCalculator(binary_sequences_p, KSTAR_RANGE, p=p)
       stats_null[p]              = p_val_info.min_pval_plus
    else:
       binary_sequences_p         = np.random.binomial(n=1, p=p  , size=(1000000, K_M))
       p_val_info                 = PValueCalculator(binary_sequences_p, KSTAR_RANGE, p=p)
       stats_null[p]              = p_val_info.min_pval_plus
       
       binary_sequences_1_m_p     = np.random.binomial(n=1, p=1-p, size=(1000000, K_M))
       p_val_info                 = PValueCalculator(binary_sequences_1_m_p, KSTAR_RANGE, 1-p)
       stats_null[1-p]              = p_val_info.min_pval_plus
              
    return stats_null

#%%        
# ----------------------------------------------------------------------
# 2) Compute the nearest neighbours
# ----------------------------------------------------------------------

def compute_nearest_neighbors(X, Y, K_M, n_jobs=10, chunk_size=1000):
    # Combine datasets A and B
    combined_dataset = np.vstack((X, Y))

    # Initialize the NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=K_M, metric='euclidean', n_jobs=n_jobs)  # n_jobs=-1 uses all CPU cores
     
    # Fit the model on the combined dataset
    nn_model.fit(combined_dataset)
    
    #to be used in IDE
    Knn_model= NearestNeighbors(n_neighbors=K_M*10, metric='euclidean', n_jobs=n_jobs)  # n_jobs=-1 uses all CPU cores
    Knn_model.fit(combined_dataset)

    # Determine chunk size for large datasets (1000 points at a time)

    total_samples = combined_dataset.shape[0]

    # Preallocate memory for all_indices
    all_indices = np.empty((total_samples, K_M), dtype=int)

    # Process the dataset in chunks if it has more than 1000 points
    if total_samples > chunk_size:
        for start_idx in range(0, total_samples, chunk_size):
            if start_idx%(chunk_size*10)==0:
                print(f"KNN completed: {start_idx/total_samples*100:.2f} %")
            end_idx = min(start_idx + chunk_size, total_samples)

            # Find the nearest neighbors for the current chunk
            indices = nn_model.kneighbors(combined_dataset[start_idx:end_idx, :])[1]

            # Assign the results directly to the preallocated array
            all_indices[start_idx:end_idx, :] = indices
    else:
        # If the dataset is small enough, process all at once
        all_indices = nn_model.kneighbors(combined_dataset)[1]

    return all_indices, Knn_model

#%%
# ----------------------------------------------------------------------
# 3) Clustering functions
# ----------------------------------------------------------------------

def cluster(data,K_M,Z=2.65):
    # Adjust maxk based on the number of samples
    data.compute_distances(maxk=K_M)
    data.compute_id_2NN()
    data.compute_density_kstarNN()
    data.compute_clustering_ADP(Z=Z, halo=False)
    return data 

def partitioning_function(X,Y,result_dict,p_ext=1e-5,Z=2.65 ):
    print("-----------------------------------------------------------------")
    print("Clustering")
    # For all points in the dataset, we will now partition them into groups with DPA clustering
    from dadapy import Data
    
    combined_dataset          = np.vstack((X, Y))
    
    # cluster X-overdensities
    combined_dataset_putative_p = np.vstack((result_dict['Upsilon_i_Y_inj'][:,np.newaxis], result_dict['Upsilon_i_Y'][:,np.newaxis]))
    
    indx_plus                 = np.where(combined_dataset_putative_p > result_dict['Upsilon_star_plus'][p_ext])[0]

    if len(indx_plus)>3:
        data_plus       = Data(combined_dataset[indx_plus], verbose=False)
        data_plus       = cluster(data_plus,min(result_dict['K_M'],len(indx_plus)-1),Z=Z)
        clusters_plus   = [indx_plus[data_plus.cluster_assignment == i] for i in range(len(data_plus.cluster_centers))]
    else:
        clusters_plus = [ list(indx_plus)] 
    # cluster Y-overdensities
    
    combined_dataset_putative_m = np.vstack((result_dict['Upsilon_i_X'][:,np.newaxis], result_dict['Upsilon_i_X_inj'][:,np.newaxis]))
    
    indx_minus                 = np.where(combined_dataset_putative_m > result_dict['Upsilon_star_minus'][p_ext])[0]

    if len(indx_minus)>3:
        data_minus       = Data(combined_dataset[indx_minus], verbose=False)
        data_minus       = cluster(data_minus,min(result_dict['K_M'],len(indx_minus)-1),Z=Z)
        clusters_minus  = [indx_minus[data_minus.cluster_assignment == i] for i in range(len(data_minus.cluster_centers))]
    else:
        clusters_minus = [ list(indx_minus)] 
        
    print("-----------------------------------------------------------------")
    return clusters_plus,clusters_minus



