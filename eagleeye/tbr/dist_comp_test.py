#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:54:52 2025

@author: sspringe
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import time

def compute_nearest_neighbors(KM, datasetA, datasetB, chunk_size=10000, n_jobs=10):
    # Combine datasets A and B
    combined_dataset = np.vstack((datasetA, datasetB))

    # Initialize the NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=KM, metric='euclidean', n_jobs=n_jobs)  # n_jobs=-1 uses all CPU cores
     
    # Fit the model on the combined dataset
    nn_model.fit(combined_dataset)
    
    #to be used in IDE
    Knn_model= NearestNeighbors(n_neighbors=KM*4, metric='euclidean', n_jobs=n_jobs)  # n_jobs=-1 uses all CPU cores
    Knn_model.fit(combined_dataset)

    # Determine chunk size for large datasets (100,000 points at a time)

    total_samples = combined_dataset.shape[0]

    # Preallocate memory for all_indices
    all_indices = np.empty((total_samples, KM), dtype=int)

    # Process the dataset in chunks if it has more than 100,000 points
    if total_samples > chunk_size:
        for start_idx in range(0, total_samples, chunk_size):
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

# Example usage:
datasetA = np.random.random((50000, 5)).astype(np.float32)  # Replace with your dataset
datasetB = np.random.random((50000, 5)).astype(np.float32)  # Replace with your dataset
KM = 500  # Example number of nearest neighbors
chunk_size = 10000
# Call the function
t = time.time()
indices, Knn_model = compute_nearest_neighbors(KM, datasetA, datasetB,chunk_size=chunk_size, n_jobs=-1 )
elapsed = time.time() - t

print(f"Computation time: {elapsed:.2f} seconds")
