#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:59:01 2024

@author: Sebastian Springer (sspringe137)
"""

import numpy as np
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
from IPython.display import display
# from annoy import AnnoyIndex


# def create_binary_array_annoy(data, num_trees=10, num_neighbors=1000):
#     """
#     Create a binary array indicating if the nearest neighbor is from the first half or second half of the dataset.

#     Args:
#     data (np.ndarray): Input data matrix of shape (n, d).
#     num_trees (int): Number of trees to build in Annoy index.
#     num_neighbors (int): Number of nearest neighbors to find.

#     Returns:
#     np.ndarray: Binary array of shape (n, num_neighbors).
#     """
#     # Creating Annoy index
#     f = data.shape[1]  # Number of features
#     t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed

#     # Adding items to the index
#     for i in range(data.shape[0]):
#         t.add_item(i, data[i])

#     # Building the index
#     t.build(num_trees)

#     # Getting nearest neighbors
#     nearest_neighbors = {}
#     for i in range(t.get_n_items()):
#         nearest_neighbors[i] = t.get_nns_by_item(i, num_neighbors)

#     # Create a binary array based on nearest neighbors
#     binary_sequences = np.zeros((data.shape[0], num_neighbors), dtype=int)
#     halfway_point = data.shape[0] // 2

#     for i, neighbors in nearest_neighbors.items():
#         binary_sequences[i] = [1 if neighbor >= halfway_point else 0 for neighbor in neighbors]

#     # Returning the binary array
#     return binary_sequences[halfway_point:,:]


def create_binary_array_cdist(mixed_samples, reference_samples, num_neighbors=1000, num_cores=10, time_series=0, partition_size = 100):
    """
    Create a binary array indicating if the nearest neighbor is from the first half or second half of the dataset.

    Args:
    mixed_samples (np.ndarray): Input test data matrix of shape (n, d).
    reference_samples (np.ndarray): Input reference data matrix of shape (n, d).
    num_neighbors (int): Number of nearest neighbors to find.
    num_cores (int): Number of cores to use for parallel processing.
    time_series (int): Number of time steps to exclude around each sample.

    Returns:
    np.ndarray: Binary array of shape (n, num_neighbors).
    np.ndarray: Neighborhood indexes of shape (n, num_neighbors).
    """
    if time_series > 0:
        adjusted_neighbors = num_neighbors + time_series
    else:
        adjusted_neighbors = num_neighbors

    D_parallel = calculate_distances_parallel(mixed_samples, reference_samples, adjusted_neighbors, num_cores,partition_size)
    binary_array_cdist_parallel, neighbourhood_indexes = process_distances(D_parallel, num_neighbors, time_series)
    return binary_array_cdist_parallel, neighbourhood_indexes



def compute_sorted_distances(samples1, samples2, num_neighbors):
    # Calculate the pairwise Euclidean distances
    distances = cdist(samples1, samples2, 'euclidean')
    
    # Get the sorted indices of distances
    sorted_indices = np.argsort(distances, axis=1)[:, :num_neighbors]
    
    # Get the sorted distances using the indices
    sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)
    
    return sorted_distances, sorted_indices


def process_partition(partition_idx, mixed_samples, reference_samples, num_neighbors, partition_size):
    start_idx = partition_size * partition_idx
    end_idx = min(partition_size * (partition_idx + 1), mixed_samples.shape[0])
    partition = mixed_samples[start_idx:end_idx, :]
    sorted_Y, sorted_idx = compute_sorted_distances(partition, reference_samples, num_neighbors)
    sorted_Y1, sorted_idx1 = compute_sorted_distances(partition, mixed_samples, num_neighbors)
    return start_idx, end_idx, sorted_Y, sorted_Y1, sorted_idx, sorted_idx1


def calculate_distances_parallel(mixed_samples, reference_samples, num_neighbors=1000, num_cores=10, partition_size = 100):
    """
    Calculate distances in parallel using the specified number of cores.

    Args:
    mixed_samples (np.ndarray): Input test data matrix of shape (n, d).
    reference_samples (np.ndarray): Input reference data matrix of shape (n, d).
    num_neighbors (int): Number of nearest neighbors to find.
    num_cores (int): Number of cores to use for parallel processing.

    Returns:
    np.ndarray: Distance matrix of shape (n, num_neighbors, 2).
    """
    num_samples = mixed_samples.shape[0]
    # partition_size = 100
    num_partitions = (num_samples + partition_size - 1) // partition_size

    sorted_Y = np.empty((num_samples, num_neighbors))
    sorted_Y1 = np.empty((num_samples, num_neighbors))
    
    sorted_idx = np.empty((num_samples, num_neighbors), dtype=int)
    sorted_idx1 = np.empty((num_samples, num_neighbors), dtype=int)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(process_partition, jj, mixed_samples, reference_samples, num_neighbors, partition_size)
            for jj in range(num_partitions)
        ]

        for future in as_completed(futures):
            start_idx, end_idx, sorted_Y_part, sorted_Y1_part, sorted_idx_part, sorted_idx1_part = future.result()
            sorted_Y[start_idx:end_idx, :] = sorted_Y_part
            sorted_Y1[start_idx:end_idx, :] = sorted_Y1_part
            sorted_idx[start_idx:end_idx, :] = sorted_idx_part
            sorted_idx1[start_idx:end_idx, :] = sorted_idx1_part
            print(f'Processing partition {start_idx // partition_size + 1}/{num_partitions}')
    
    loc1 = np.concatenate((sorted_Y, sorted_Y1), axis=1)
    loc2 = np.concatenate((np.zeros(sorted_Y.shape), np.ones(sorted_Y1.shape)), axis=1)
    loc3 = np.concatenate((sorted_idx, sorted_idx1), axis=1)
    
    D = np.zeros((loc2.shape[0], loc2.shape[1], 3))
    D[:, :, 0] = loc1
    D[:, :, 1] = loc2
    D[:, :, 2] = loc3

    return D


def process_distances(D, num_neighbors=1000, time_series=0):
    
    # Assign infinity to distances within the time_series range
    if time_series > 0:
        for i in range(D.shape[0]):
            lower_bound = max(0, i - time_series)
            upper_bound = min(D.shape[0], i + time_series)
    
            # Create a mask to identify indices to exclude
            exclusion_mask = (((D[i,:,2] >= lower_bound) & (D[i,:,2] <= upper_bound))*D[i,:,1]).astype(bool)
    
            # Set distances to infinity where the exclusion mask is True
            D[i, exclusion_mask, 0] = np.inf
        
    sorting_indices = np.argsort(D[:, :, 0], axis=1)
    neighbourhood_indexes = np.take_along_axis(D[:, :, 2], sorting_indices, axis=1) 
    sorted_D_1 = np.take_along_axis(D[:, :, 1], sorting_indices, axis=1)
    binary_sequences = sorted_D_1[:, :num_neighbors].copy()
    return binary_sequences,neighbourhood_indexes


