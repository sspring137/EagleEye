import numpy as np
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed

###############################################################################
# 1) Helper: compute_sorted_distances (same as in your original code)
###############################################################################
def compute_sorted_distances(samples1, samples2, num_neighbors):
    """
    Compute pairwise distances between samples1 and samples2,
    then return (top-distances, top-indices) of shape (len(samples1), num_neighbors).
    """
    dist_matrix = cdist(samples1, samples2, 'euclidean')
    sorted_idx  = np.argsort(dist_matrix, axis=1)[:, :num_neighbors]
    # Gather the corresponding distances
    row_idx     = np.arange(dist_matrix.shape[0])[:, None]
    sorted_dists = dist_matrix[row_idx, sorted_idx]
    return sorted_dists, sorted_idx


###############################################################################
# 2) Parallel partition function: process_partition_subset
###############################################################################
def process_partition_subset(
    partition_idx,
    subset_indices,
    mixed_samples,
    reference_samples,
    num_neighbors,
    partition_size
):
    """
    Worker function that handles one partition of the subset rows in mixed_samples.

    It returns:
        start_local, end_local,
        sorted_Y_ref, sorted_Y_mixed,
        sorted_idx_ref, sorted_idx_mixed
    where:
      - sorted_Y_ref, sorted_idx_ref => top num_neighbors from reference_samples
      - sorted_Y_mixed, sorted_idx_mixed => top num_neighbors from *all* mixed_samples
    """
    start_local = partition_idx * partition_size
    end_local   = min((partition_idx + 1) * partition_size, len(subset_indices))
    
    # The actual rows in mixed_samples for this partition
    actual_indices  = subset_indices[start_local:end_local]
    partition_data  = mixed_samples[actual_indices]  # shape: (p, d)
    
    # 1) Distances to reference_samples
    sorted_dists_ref, sorted_idx_ref = compute_sorted_distances(
        partition_data, reference_samples, num_neighbors
    )
    # 2) Distances to the entire mixed_samples
    sorted_dists_mix, sorted_idx_mix = compute_sorted_distances(
        partition_data, mixed_samples, num_neighbors
    )
    
    return (start_local, end_local,
            sorted_dists_ref, sorted_dists_mix,
            sorted_idx_ref,  sorted_idx_mix)


###############################################################################
# 3) Main parallel distance function: calculate_distances_subset_parallel
###############################################################################
def calculate_distances_subset_parallel(
    mixed_samples,
    reference_samples,
    subset_indices,
    num_neighbors=1000,
    num_cores=10,
    partition_size=100
):
    """
    Similar to your original calculate_distances_parallel, but only for a
    subset of rows in mixed_samples (given by subset_indices).

    Returns a D array of shape:
        (len(subset_indices), num_neighbors * 2, 3)

    Where D[i, :, 0] = distances,
          D[i, :, 1] = 0 or 1 (0 => from reference, 1 => from mixed),
          D[i, :, 2] = indices in whichever array it belongs to.
    """
    num_samples_subset = len(subset_indices)
    num_partitions     = (num_samples_subset + partition_size - 1) // partition_size

    # Allocate final D
    D = np.empty((num_samples_subset, num_neighbors * 2, 3), dtype=np.float32)

    def process_and_store_subset(future):
        (start_local, end_local,
         sorted_dists_ref, sorted_dists_mix,
         sorted_idx_ref,  sorted_idx_mix) = future.result()

        # Concatenate the two sets of distances
        loc_dists = np.concatenate([sorted_dists_ref, sorted_dists_mix], axis=1)  # shape (p, 2*num_neighbors)
        # Tag array (0 => reference, 1 => mixed)
        tag_ref   = np.zeros_like(sorted_dists_ref,  dtype=np.float32)
        tag_mix   = np.ones_like(sorted_dists_mix,   dtype=np.float32)
        loc_tags  = np.concatenate([tag_ref, tag_mix], axis=1)  # shape (p, 2*num_neighbors)
        # Indices
        loc_idx   = np.concatenate([sorted_idx_ref, sorted_idx_mix], axis=1)      # shape (p, 2*num_neighbors)

        # Store into D
        D[start_local:end_local, :, 0] = loc_dists
        D[start_local:end_local, :, 1] = loc_tags
        D[start_local:end_local, :, 2] = loc_idx

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for partition_idx in range(num_partitions):
            fut = executor.submit(
                process_partition_subset,
                partition_idx,
                subset_indices,
                mixed_samples,
                reference_samples,
                num_neighbors,
                partition_size
            )
            futures.append(fut)

        for fut in as_completed(futures):
            process_and_store_subset(fut)

    return D


###############################################################################
# 4) Helper: process_distances (same as in your code)
###############################################################################
def process_distances(D, num_neighbors=1000, time_series=0):
    """
    Sort each row by distance, optionally exclude neighbors within a time_series window,
    then keep only the first num_neighbors columns.

    Returns:
        binary_sequences: shape (n, num_neighbors) with 0/1
        neighbourhood_indexes: shape (n, num_neighbors)
    """

    # If time_series > 0, optionally set distances to inf for neighbors in [i-time_series, i+time_series]
    if time_series > 0:
        for i in range(D.shape[0]):
            lower_bound = max(0, i - time_series)
            upper_bound = min(D.shape[0], i + time_series)
            # Exclude neighbors that are within that range *and* from the same array
            # (In some use-cases, you'd check D[i, :, 2] for neighbor index.)
            exclusion_mask = (
                ((D[i, :, 2] >= lower_bound) & (D[i, :, 2] <= upper_bound)) 
                * (D[i, :, 1] == 1.0)  # if you're excluding only within 'mixed' or something
            ).astype(bool)
            D[i, exclusion_mask, 0] = np.inf

    # Sort by the distance dimension D[:, :, 0]
    sorting_indices        = np.argsort(D[:, :, 0], axis=1)
    # Reorder the neighbor indexes
    neighbourhood_indexes  = np.take_along_axis(D[:, :, 2], sorting_indices, axis=1)
    # Reorder the 0/1 tags
    sorted_tags            = np.take_along_axis(D[:, :, 1], sorting_indices, axis=1)

    # Keep only the top num_neighbors
    binary_sequences       = sorted_tags[:, :num_neighbors].astype(bool)  # 0 => False, 1 => True
    neighbourhood_indexes  = neighbourhood_indexes[:, :num_neighbors]

    return binary_sequences, neighbourhood_indexes


###############################################################################
# 5) Final function: create_binary_array_cdist_post_subset
###############################################################################
def create_binary_array_cdist_post_subset(
    mixed_samples,
    reference_samples,
    subset_indices,
    num_neighbors=1000,
    num_cores=10,
    validation=None,
    partition_size=100,
    time_series=0
):
    """
    Similar to create_binary_array_cdist_post, but we only compute nearest neighbors
    for the subset of rows in `mixed_samples` specified by subset_indices.

    Returns (binary_array_cdist_parallel, neighbourhood_indexes):
      - binary_array_cdist_parallel: shape (len(subset_indices), num_neighbors) => 0/1
      - neighbourhood_indexes: shape (len(subset_indices), num_neighbors) => indices in whichever array

    Args:
        mixed_samples (np.ndarray): shape (N, d)
        reference_samples (np.ndarray): shape (M, d)
        subset_indices (array-like): which rows of mixed_samples to process
        num_neighbors (int): number of nearest neighbors to keep from reference + from mixed
        num_cores (int): parallel processes
        validation (int or list or None): optional validation logic (simple version)
        partition_size (int): how many subset rows to handle per parallel chunk
        time_series (int): optional. Exclude neighbors within +/- time_series if desired.

    """
    # --------------------------------------
    # Step B: Calculate Distances (Subset)
    # --------------------------------------
    # This yields D of shape (len(subset_indices), 2*num_neighbors, 3)
    D_subset = calculate_distances_subset_parallel(
        mixed_samples       = mixed_samples,
        reference_samples   = reference_samples,
        subset_indices      = subset_indices,
        num_neighbors       = num_neighbors,
        num_cores           = num_cores,
        partition_size      = partition_size
    )

    # --------------------------------------
    # Step C: Process Distances
    # --------------------------------------
    binary_array_cdist_parallel, neighbourhood_indexes = process_distances(
        D_subset, num_neighbors=num_neighbors, time_series=time_series
    )

    # (Optional) In your original code, you truncated neighbourhood_indexes to 
    # match the number of columns in the binary array:
    # neighbourhood_indexes = neighbourhood_indexes[:, : binary_array_cdist_parallel.shape[1]]
    # But we already do that inside process_distances. So it's consistent.

    return binary_array_cdist_parallel, neighbourhood_indexes

