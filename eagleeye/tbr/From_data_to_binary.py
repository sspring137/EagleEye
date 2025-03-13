#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:59:01 2024

@author: Sebastian Springer (sspringe137)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:18:50 2024

Authors: Sebastian Springer (sspringe137), Andre Scaffidi (AndreScaffidi) and Alessandro Laio

--------------------------------------------------------------------------------
## Non-Commercial Academic and Research License (NCARL) v1.0

### Terms and Conditions

1. **Grant of License**: Permission is hereby granted, free of charge, to any person or organization obtaining a copy of this software to use, copy, modify, and distribute the software for academic research, educational purposes, and personal non-commercial projects, subject to the following conditions:

2. **Non-Commercial Use**: Non-commercial use includes any use that is not intended for or directed towards commercial advantage or monetary compensation. Examples include academic research, teaching, and personal experimentation.

3. **Acknowledgment**: Any publications or products that use the software must include the following acknowledgment:
   - "This software uses EagleEye developed by Sebastian Springer, Alessandro Laio and Andre Scaffidi at the International School for Advanced Studies (SISSA), Via Bonomea, 265, 34136 Trieste TS (Italy)."

4. **Modification and Distribution**: Users are allowed to modify and distribute the software for non-commercial purposes, provided they include this license with any distribution and acknowledge the original authors.

5. **No Warranty**: The software is provided "as-is" without any warranty of any kind.

### Contact Information

For commercial licensing, please contact Sebastian Springer at sebastian.springer@sissa.it (doc.sebastian.springer@gmail.com).
--------------------------------------------------------------------------------

## Commercial License Agreement (CLA) v1.0

### Terms and Conditions

1. **Grant of License**: Permission is hereby granted to any person or organization obtaining a copy of this software for commercial use, provided they comply with the terms and conditions outlined in this agreement and pay the applicable licensing fees.

2. **Commercial Use Definition**: Commercial use includes any use intended for or directed towards commercial advantage or monetary compensation. This includes, but is not limited to, use in a commercial product, offering services with the software, or using the software in a revenue-generating activity.

3. **Licensing Fees**: The licensee agrees to negotiate and pay a licensing fee for commercial use of the software. 

4. **Modification and Distribution**: Users are allowed to modify and distribute the software under the terms of this commercial license, provided they include this license with any distribution and acknowledge the original authors.

5. **Warranty**: The software is provided with a limited warranty as outlined in the commercial licensing agreement. Details of the warranty can be provided upon request.

### Contact Information

For licensing fees, terms, and support, please contact Sebastian Springer at sebastian.springer@sissa.it (doc.sebastian.springer@gmail.com).
--------------------------------------------------------------------------------
"""
import numpy as np
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
from IPython.display import display

def create_binary_array_cdist_post(mixed_samples, reference_samples, num_neighbors=1000, num_cores=10,validation=None,partition_size=100):
    """
    Create a binary array indicating if the nearest neighbor is from the first half or second half of the dataset.

    Args:
    mixed_samples (np.ndarray): Input test data matrix of shape (n, d).
    reference_samples (np.ndarray): Input reference data matrix of shape (n, d).
    num_neighbors (int): Number of nearest neighbors to find.
    validation (np.ndarray or int): Indicies of validation set, or size of validation set you want to use. If int, will take first n samples of reference as validation set.
    partition_size (int): Size of the partition to be processed in parallel.

    Returns:
    np.ndarray: Binary array of shape (n, num_neighbors).
    """


    D_parallel                                         = calculate_distances_parallel(mixed_samples, reference_samples, num_neighbors, num_cores,partition_size=partition_size)
    binary_array_cdist_parallel, neighbourhood_indexes = process_distances(D_parallel, num_neighbors)
    neighbourhood_indexes = neighbourhood_indexes[:,:binary_array_cdist_parallel.shape[1]]


    return binary_array_cdist_parallel, neighbourhood_indexes


def create_binary_array_cdist(mixed_samples, reference_samples, num_neighbors=1000, num_cores=10,validation=None,partition_size=100):
    """
    Create a binary array indicating if the nearest neighbor is from the first half or second half of the dataset.

    Args:
    mixed_samples (np.ndarray): Input test data matrix of shape (n, d).
    reference_samples (np.ndarray): Input reference data matrix of shape (n, d).
    num_neighbors (int): Number of nearest neighbors to find.
    validation (np.ndarray or int): Indicies of validation set, or size of validation set you want to use. If int, will take first n samples of reference as validation set.
    partition_size (int): Size of the partition to be processed in parallel.

    Returns:
    np.ndarray: Binary array of shape (n, num_neighbors).
    """

    # Decice if we need to add validation samples to the mixed_samples:
    # The below is effectively injecting one validation sample at time, but for computational purposes we 
    # can inject all validation samples at once and just adjust the neighbourhood indexes accordingly.
    if validation is not None: 
        val_size = validation if isinstance(validation, int) else len(validation)
        if val_size <= (len(reference_samples)-num_neighbors):
            if isinstance(validation, int):
                print("Validation size is: ", validation)
                mixed_samples             = np.concatenate((mixed_samples, reference_samples[:val_size,:] ), axis=0) # Stack validation samples on the bottom!!!
                reference_samples         = reference_samples[val_size:,:]
            else :
                print("Validation index list size is: ", len(validation))
                validation_samples        = reference_samples[validation,:]
                mixed_samples             = np.concatenate((mixed_samples, validation_samples ), axis=0) # Stack validation samples on the bottom!!!
                reference_samples         = np.delete(reference_samples, validation, axis=0)
            # Calculate! 
            D_parallel                                         = calculate_distances_parallel(mixed_samples, reference_samples, num_neighbors, num_cores,partition_size=partition_size)
            binary_array_cdist_parallel, neighbourhood_indexes = process_distances(D_parallel, num_neighbors)
            # Lastly, adjust the neighbourhood indexes if validation is not None
            if validation is not None:
                for index, row in enumerate(binary_array_cdist_parallel):
                    neigh                              = (neighbourhood_indexes[:,:num_neighbors][index]>=int(len(mixed_samples)-val_size))
                    binary_array_cdist_parallel[index,(neigh) & (row==True)] = 0
                    binary_array_cdist_parallel[:,0]                         =  1
            return binary_array_cdist_parallel


        elif val_size > (len(reference_samples)-num_neighbors):
            # print("Warning: Validation size is larger than size of reference - Kmax. Partianing runs!")
            if isinstance(validation, int):
                val_idx = np.array(list(range(val_size)))
            else:
                val_idx = validation

            print("Validation size is: ", val_size)


            val_idx_ch1 = val_idx[:int(val_size/2)]
            val_idx_ch2 = val_idx[int(val_size/2):]

            # Maunal partitioning of validation (only two partitions)
  
            validation_samples_ch1        = reference_samples[val_idx_ch1,:]
            mixed_samples_ch1             = np.concatenate((mixed_samples, validation_samples_ch1 ), axis=0) # Stack validation samples on the bottom!!!
            reference_samples_ch1         = np.delete(reference_samples, val_idx_ch1, axis=0)
            D_parallel                    = calculate_distances_parallel(mixed_samples_ch1, reference_samples_ch1, num_neighbors, num_cores,partition_size=partition_size)
            binary_array_cdist_parallel_ch1, neighbourhood_indexes = process_distances(D_parallel, num_neighbors)
            for index, row in enumerate(binary_array_cdist_parallel_ch1):
                neigh                              = (neighbourhood_indexes[:,:num_neighbors][index]>=int(len(mixed_samples_ch1)-len(val_idx_ch1)))
                binary_array_cdist_parallel_ch1[index,(neigh) & (row==True)] = 0
                binary_array_cdist_parallel_ch1[:,0]                         = 1
            del(D_parallel)
            del(neighbourhood_indexes)


            # Second chunk


            validation_samples_ch2        = reference_samples[val_idx_ch2,:]
            mixed_samples_ch2             = np.concatenate((mixed_samples, validation_samples_ch2 ), axis=0) # Stack validation samples on the bottom!!!
            reference_samples_ch2         = np.delete(reference_samples, val_idx_ch2, axis=0)
            D_parallel                    = calculate_distances_parallel(mixed_samples_ch2, reference_samples_ch2, num_neighbors, num_cores,partition_size=partition_size)

            binary_array_cdist_parallel_ch2, neighbourhood_indexes = process_distances(D_parallel, num_neighbors)

            for index, row in enumerate(binary_array_cdist_parallel_ch2):
                neigh                              = (neighbourhood_indexes[:,:num_neighbors][index]>=int(len(mixed_samples_ch2)-len(val_idx_ch2)))
                binary_array_cdist_parallel_ch2[index,(neigh) & (row==True)] = 0
                binary_array_cdist_parallel_ch2[:,0]                         =  1

            del(D_parallel)
            del(neighbourhood_indexes)


            binary_array_cdist_parallel = np.concatenate((binary_array_cdist_parallel_ch1, binary_array_cdist_parallel_ch2[-len(val_idx_ch2):]), axis=0)


    else:
        D_parallel                                         = calculate_distances_parallel(mixed_samples, reference_samples, num_neighbors, num_cores,partition_size=partition_size)
        binary_array_cdist_parallel, neighbourhood_indexes = process_distances(D_parallel, num_neighbors)

    # # Calculate! 
    # D_parallel                                         = calculate_distances_parallel(mixed_samples, reference_samples, num_neighbors, num_cores,partition_size=partition_size)
    # binary_array_cdist_parallel, neighbourhood_indexes = process_distances(D_parallel, num_neighbors)

    # # Lastly, adjust the neighbourhood indexes if validation is not None
    # if validation is not None:
    #     for index, row in enumerate(binary_array_cdist_parallel):
    #         neigh                              = (neighbourhood_indexes[:,:num_neighbors][index]>=int(len(mixed_samples)-val_size))
    #         binary_array_cdist_parallel[index,(neigh) & (row==True)] = 0
    #         binary_array_cdist_parallel[:,0]                         =  1

    return binary_array_cdist_parallel




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


# OLD MEMONRY INTENSIVE
# def calculate_distances_parallel(mixed_samples, reference_samples, num_neighbors=1000, num_cores=10, partition_size = 100):
#     """
#     Calculate distances in parallel using the specified number of cores.

#     Args:
#     mixed_samples (np.ndarray): Input test data matrix of shape (n, d).
#     reference_samples (np.ndarray): Input reference data matrix of shape (n, d).
#     num_neighbors (int): Number of nearest neighbors to find.
#     num_cores (int): Number of cores to use for parallel processing.

#     Returns:
#     np.ndarray: Distance matrix of shape (n, num_neighbors, 2).
#     """
#     num_samples = mixed_samples.shape[0]
#     # partition_size = 100
#     num_partitions = (num_samples + partition_size - 1) // partition_size

#     sorted_Y = np.empty((num_samples, num_neighbors))
#     sorted_Y1 = np.empty((num_samples, num_neighbors))
    
#     sorted_idx = np.empty((num_samples, num_neighbors), dtype=int)
#     sorted_idx1 = np.empty((num_samples, num_neighbors), dtype=int)

#     with ProcessPoolExecutor(max_workers=num_cores) as executor:
#         futures = [
#             executor.submit(process_partition, jj, mixed_samples, reference_samples, num_neighbors, partition_size)
#             for jj in range(num_partitions)
#         ]

#         for future in as_completed(futures):
#             start_idx, end_idx, sorted_Y_part, sorted_Y1_part, sorted_idx_part, sorted_idx1_part = future.result()
#             sorted_Y[start_idx:end_idx, :] = sorted_Y_part
#             sorted_Y1[start_idx:end_idx, :] = sorted_Y1_part
#             sorted_idx[start_idx:end_idx, :] = sorted_idx_part
#             sorted_idx1[start_idx:end_idx, :] = sorted_idx1_part
#             print(f'Processing partition {start_idx // partition_size + 1}/{num_partitions}')
    
#     loc1 = np.concatenate((sorted_Y, sorted_Y1), axis=1)
#     loc2 = np.concatenate((np.zeros(sorted_Y.shape), np.ones(sorted_Y1.shape)), axis=1)
#     loc3 = np.concatenate((sorted_idx, sorted_idx1), axis=1)
    
#     del(sorted_Y)
#     del(sorted_Y1)
#     del(sorted_idx)
#     del(sorted_idx1)

#     D = np.empty((loc2.shape[0], loc2.shape[1], 3), dtype=np.float32)
#     D[:, :, 0] = loc1
#     D[:, :, 1] = loc2
#     D[:, :, 2] = loc3

#     return D



def calculate_distances_parallel(mixed_samples, reference_samples, num_neighbors=1000, num_cores=10, partition_size=100):
    """
    Calculate distances in parallel using the specified number of cores in a memory-efficient way.

    Args:
    mixed_samples (np.ndarray): Input test data matrix of shape (n, d).
    reference_samples (np.ndarray): Input reference data matrix of shape (m, d).
    num_neighbors (int): Number of nearest neighbors to find.
    num_cores (int): Number of cores to use for parallel processing.

    Returns:
    np.ndarray: Distance matrix of shape (n, num_neighbors * 2, 3).
    """
    num_samples = mixed_samples.shape[0]
    num_partitions = (num_samples + partition_size - 1) // partition_size

    # Initialize an empty array for the final results
    D = np.empty((num_samples, num_neighbors * 2, 3), dtype=np.float32)

    def process_and_store(future):
        start_idx, end_idx, sorted_Y_part, sorted_Y1_part, sorted_idx_part, sorted_idx1_part = future.result()
        # Concatenate results for this partition
        loc1 = np.concatenate((sorted_Y_part, sorted_Y1_part), axis=1)
        loc2 = np.concatenate((np.zeros(sorted_Y_part.shape), np.ones(sorted_Y1_part.shape)), axis=1)
        loc3 = np.concatenate((sorted_idx_part, sorted_idx1_part), axis=1)
        # Store directly into D to save memory
        D[start_idx:end_idx, :, 0] = loc1
        D[start_idx:end_idx, :, 1] = loc2
        D[start_idx:end_idx, :, 2] = loc3
        #print(f'Processing partition {start_idx // partition_size + 1}/{num_partitions}')

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(process_partition, jj, mixed_samples, reference_samples, num_neighbors, partition_size)
            for jj in range(num_partitions)
        ]

        for future in as_completed(futures):
            process_and_store(future)

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


