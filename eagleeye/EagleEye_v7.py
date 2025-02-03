#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:15:36 2025

@author: johan
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from dadapy import Data

# If you have custom modules, adjust sys.path or imports as needed:
# module_path = '../../eagleeye'
# sys.path.append(module_path)
# import From_data_to_binary_post  # e.g., if used
import From_data_to_binary


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

# ----------------------------------------------------------------------
# 1) PValueCalculator Class
# ----------------------------------------------------------------------

class PValueCalculator:
    """
    A class that computes for each row in a binary array the negative-log
    p-values (minus and plus tails) for various subsequence lengths k,
    and finds the row-wise maxima.
    """
    def __init__(self, 
                 binary_sequence, 
                 kstar_range, 
                 p=0.5, 
                 pvals_dict=None,
                 verbose=False):
        """
        Parameters
        ----------
        binary_sequence : np.ndarray of shape (N, M)
            Binary (0/1) array of data (N rows, M columns).
        kstar_range : iterable of ints
            A list (or range) of subsequence lengths k for which we compute
            negative-log p-values.
        p : float, optional
            Probability parameter for the Binomial distribution (default=0.5).
        pvals_dict : dict or None, optional
            Optional precomputed dict:
                pvals_dict[k][stat_sum] = (neg_log_minus, neg_log_plus).
            If None, we'll compute it ourselves.
        """
        self.binary_sequence = binary_sequence
        self.kstar_range = list(kstar_range)
        self.p = p
        self.verbose = verbose

        # 1) Cumulative sums for fast row-sums of first k bits
        self.binary_cumsum = np.c_[
            np.zeros((binary_sequence.shape[0], 1), dtype=int),
            np.cumsum(binary_sequence, axis=1)
        ]

        # 2) Build (or reuse) the dictionary of negative-log p-values
        if pvals_dict is not None:
            self.pvals_dict = pvals_dict
        else:
            self.pvals_dict = self.build_pvals_dict()

        # 3) For each k in kstar_range, compute an array of neg_log_minus/neg_log_plus
        self.pval_array_dict = {}
        for kstar in self.kstar_range:
            self.pval_array_dict[kstar] = self.compute_neglog_array(kstar)
            if self.verbose:
                print(f"Computed p-values for k={kstar}")
        # 4) Compute row-wise maxima across kstar_range
        self.compute_rowwise_maxima()

    def build_pvals_dict(self):
        """
        Build a dictionary that, for each k in kstar_range and
        each stat_sum in [0..k], stores (neg_log_minus, neg_log_plus).

        Returns
        -------
        dict
            pvals_dict[k][i] = (neg_log_minus, neg_log_plus).
        """
        pvals_dict = {}
        for k in self.kstar_range:
            # CDF arrays for minus & plus tails
            cdf_array_minus = np.cumsum( 
                binom_pmf_range(k, self.p) 
            )  # cdf 
            cdf_array_plus = np.cumsum(
                binom_pmf_range(k, self.p)[::-1]
            )[::-1]  # 1-cdf 

            pvals_dict[k] = {}
            for i in range(k + 1):
                neg_log_minus = -np.log(cdf_array_minus[i])
                neg_log_plus  = -np.log(cdf_array_plus[i])
                pvals_dict[k][i] = (neg_log_minus, neg_log_plus)
        return pvals_dict

    def compute_neglog_array(self, kstar):
        """
        For each row, sum up the first kstar bits, then look up neg_log_minus
        and neg_log_plus in self.pvals_dict.

        Returns
        -------
        neglog_array : np.ndarray of shape (N, 2)
            [neg_log_minus, neg_log_plus] for each row.
        """
        stat_sum_list = self.binary_cumsum[:, kstar]  # sums of first kstar bits
        N = len(stat_sum_list)
        neglog_array = np.zeros((N, 2), dtype=float)

        # Cache repeated lookups
        unique_sums = np.unique(stat_sum_list)
        lookup_cache = {
            val: self.pvals_dict[kstar][val] for val in unique_sums
        }

        for i in range(N):
            val = stat_sum_list[i]
            neglog_array[i, :] = lookup_cache[val]

        return neglog_array

    def compute_rowwise_maxima(self):
        """
        Create arrays with the maximum neg_log_minus / neg_log_plus per row
        and track which kstar attains those maxima.

        Attributes
        ----------
        self.min_pval_minus : np.ndarray of shape (N,)
            Maximum -log(p_minus) among all k for each row.
        self.kstar_min_pval_minus : np.ndarray of shape (N,)
            The kstar that attains the maximum -log(p_minus) per row.
        self.min_pval_plus : np.ndarray of shape (N,)
            Maximum -log(p_plus) among all k for each row.
        self.kstar_min_pval_plus : np.ndarray of shape (N,)
            The kstar that attains the maximum -log(p_plus) per row.
        """
        N = self.binary_sequence.shape[0]
        K = len(self.kstar_range)

        neg_log_minus_matrix = np.zeros((N, K), dtype=float)
        neg_log_plus_matrix  = np.zeros((N, K), dtype=float)

        for col_idx, k in enumerate(self.kstar_range):
            neglog_arr = self.pval_array_dict[k]  # (N, 2)
            neg_log_minus_matrix[:, col_idx] = neglog_arr[:, 0]
            neg_log_plus_matrix[:,  col_idx] = neglog_arr[:, 1]

        # Row-wise maximum and index
        self.min_pval_minus = np.max(neg_log_minus_matrix, axis=1)
        argmax_minus = np.argmax(neg_log_minus_matrix, axis=1)
        self.min_pval_plus = np.max(neg_log_plus_matrix, axis=1)
        argmax_plus  = np.argmax(neg_log_plus_matrix, axis=1)

        kstar_array = np.array(self.kstar_range)
        self.kstar_min_pval_minus = kstar_array[argmax_minus]
        self.kstar_min_pval_plus  = kstar_array[argmax_plus]

# ----------------------------------------------------------------------
# 2) Helper to compute binomial pmf for range(k+1)
#    (We replicate logic from scipy.stats.binom for clarity/compatibility)
# ----------------------------------------------------------------------
from math import comb, log

def binom_pmf_range(k, p):
    """
    Returns an array of length (k+1) with binomial PMF for n=k, p=p, x=0..k.
    """
    return np.array([comb(k, x) * (p**x) * ((1-p)**(k-x)) for x in range(k+1)])


# ----------------------------------------------------------------------
# 3) calculate_p_values function
# ----------------------------------------------------------------------

def calculate_p_values(binary_sequence, kstar_range, p, validation=None,verbose=False):
    """
    Computes the -log p-values (minus and plus) for each row in binary_sequence
    over the given kstar_range. Returns a dictionary of summary statistics.

    If 'validation' is not None (int or list of indices), the final arrays
    are split into training and validation subsets.
    """
    p_val_info = PValueCalculator(binary_sequence, kstar_range, p=p, pvals_dict=None,verbose=verbose)

    Upsilon_i_minus = p_val_info.min_pval_minus
    Upsilon_i_plus  = p_val_info.min_pval_plus

    statistics = {
        'Upsilon_i_minus': [],
        'kstar_minus': [],
        'Upsilon_i_plus': [],
        'kstar_plus': [],
        'Upsilon_i_Val_minus': [],
        'kstar_Val_minus': [],
        'Upsilon_i_Val_plus': [],
        'kstar_Val_plus': []
    }

    if validation is not None:
        # Determine how many entries go to "validation"
        if isinstance(validation, int):
            len_val = validation
        else:
            len_val = len(validation)

        # Slice off the last len_val entries as "validation"
        statistics['Upsilon_i_Val_minus'] = Upsilon_i_minus[-len_val:]
        statistics['kstar_Val_minus']     = p_val_info.kstar_min_pval_minus[-len_val:]
        statistics['Upsilon_i_Val_plus']  = Upsilon_i_plus[-len_val:]
        statistics['kstar_Val_plus']      = p_val_info.kstar_min_pval_plus[-len_val:]

        statistics['Upsilon_i_minus'] = Upsilon_i_minus[:-len_val]
        statistics['kstar_minus']     = p_val_info.kstar_min_pval_minus[:-len_val]
        statistics['Upsilon_i_plus']  = Upsilon_i_plus[:-len_val]
        statistics['kstar_plus']      = p_val_info.kstar_min_pval_plus[:-len_val]

    else:
        statistics['Upsilon_i_minus'] = Upsilon_i_minus
        statistics['kstar_minus']     = p_val_info.kstar_min_pval_minus
        statistics['Upsilon_i_plus']  = Upsilon_i_plus
        statistics['kstar_plus']      = p_val_info.kstar_min_pval_plus

    return statistics

# ----------------------------------------------------------------------
# 4) Anomaly partitioning functions
# ----------------------------------------------------------------------

def partition_rows_with_threshold(arr, threshold=0.1):
    """
    Partitions the row indices of `arr` into sets of indices where each set
    contains rows that share at least `threshold * K` elements 
    (directly or transitively).
    """
    N, K = arr.shape
    required_common = math.ceil(threshold * K)

    parent = list(range(N))  # Union-Find parent list

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Convert each row to a set of elements for intersection checks
    row_sets = [set(arr[i]) for i in range(N)]

    # O(N^2) approach: union rows that share >= required_common
    for i in range(N):
        for j in range(i + 1, N):
            if len(row_sets[i].intersection(row_sets[j])) >= required_common:
                union(i, j)

    # Group by final representative
    partitions_dict = defaultdict(list)
    for i in range(N):
        r = find(i)
        partitions_dict[r].append(i)

    # Convert to list of lists
    partitions = list(partitions_dict.values())
    partition_final = partition_rows_final_step(arr, partitions)
    return partition_final

def partition_rows_final_step(arr, partitions):
    """
    Helper to map partition row indices to the first column of arr 
    (assuming arr[:,0] are the actual IDs).
    """
    partition_final = []
    for grp in partitions:
        partition_final.append(arr[list(grp), 0])
    return partition_final

def collect_unique_until_greater(array_2d: np.ndarray, threshold: int) -> np.ndarray:
    """
    For each row, collect elements until encountering a value > threshold.
    Return unique values of all collected elements across rows.
    """
    if array_2d.size == 0:
        return np.array([], dtype=array_2d.dtype)

    slices = []
    for row in array_2d:
        mask = row > threshold
        idx = np.argmax(mask)
        if mask[idx]:
            # Found at least one element > threshold
            # Take elements up to idx-1
            slices.append(row[:idx-1])
        else:
            # No elements in this row exceed threshold
            slices.append(row)

    if slices:
        all_collected = np.concatenate(slices)
        return np.unique(all_collected)
    else:
        return np.array([], dtype=array_2d.dtype)

def exclude_and_take_first_k(A: np.ndarray,
                            EXCL: list,
                            K_M: int,
                            fill_value=np.infty) -> np.ndarray:
    """
    For each row of A, exclude any elements in EXCL, then take the first K_M 
    remaining elements (preserving order). The rest are filled with fill_value.
    """
    n_rows, n_cols = A.shape
    result = np.full((n_rows, K_M), fill_value=fill_value, dtype=A.dtype)
    excluded_mask = np.isin(A, EXCL)

    for i in range(n_rows):
        valid_indices = np.where(~excluded_mask[i])[0]
        chosen_indices = valid_indices[:K_M]
        num_chosen = len(chosen_indices)
        result[i, :num_chosen] = A[i, chosen_indices]

    return result

# ----------------------------------------------------------------------
# 5) Cleaned iterative_equalization
#    (Originally from postprocessing_equalization.py)
# ----------------------------------------------------------------------

def iterative_equalization(
    unique_elements_dict,
    dataset_T, 
    dataset_R, 
    Upsilon_i_T_wrt_R, 
    Upsilon_star_plus,
    K_M,
    NUMBER_CORES,
    PARTITION_SIZE,
):
    """
    Iteratively equalize overdensities of dataset_T wrt dataset_R.

    Parameters
    ----------
    dataset_T : np.ndarray
        Target dataset whose overdensities we want to iteratively remove.
    dataset_R : np.ndarray
        Reference dataset used to calculate overdensities.
    Upsilon_i_T_wrt_R : np.ndarray
        Array of row-wise -log p-values (or similar measure of anomaly) 
        for dataset_T with respect to dataset_R.
    Upsilon_star_plus : float, np.ndarray
        Threshold/s above which a row is considered an overdensity.
    K_M : int
        Number of neighbors used in building the binary sequences.
    NUMBER_CORES : int
        Number of CPU cores to use for parallel operations.
    PARTITION_SIZE : int
        Batch size or partition size for large datasets (used in binary array creation).

    Returns
    -------
    unique_elements : list
        The final list of row indices in dataset_T identified as anomalies 
        (or “overdensities”) to remove.
    """
    # Identify points that exceed the threshold
    subset_indices = np.where(Upsilon_i_T_wrt_R > Upsilon_star_plus)[0]
    p = len(dataset_T) / (len(dataset_T) + len(dataset_R))
    # Create a (larger) binary array to have more room for iteration
    # (Requires your custom function from From_data_to_binary_post)
    # Check if the smallest value of Upsilon_star_plus is less than the smallest key in unique_elements_dict


    if len(unique_elements_dict.keys()) == 0: 
        min_UE = np.inf
    else:
        min_UE = min(unique_elements_dict.keys())

    if Upsilon_star_plus < min_UE:
        print("The smallest value of Upsilon_star_plus is less than the smallest key in unique_elements_dict.")

        already_computed_         = [item for item in unique_elements_dict.items()]
        subset_indices            = [x for x in subset_indices if x not in already_computed_]

        import From_data_to_binary_post
        binary_sequences_pp, neighborhood_idx_pp = From_data_to_binary_post.create_binary_array_cdist_post_subset(
            dataset_T,
            dataset_R,
            subset_indices,
            num_neighbors=K_M * 4,
            num_cores=NUMBER_CORES,
            validation=None,
            partition_size=PARTITION_SIZE
        )

        # label anomalies
        label_anomalies = (Upsilon_i_T_wrt_R > Upsilon_star_plus).astype(int)
        list_temp = np.where(label_anomalies)[0]
        Upsilon_i_temp = Upsilon_i_T_wrt_R[list_temp].copy()
        key_thresh = Upsilon_i_temp.max()
        # Find the row(s) with highest Upsilon
        index_max = np.where(Upsilon_i_temp == Upsilon_i_temp.max())[0]



        # Shift indexes so that if a row is '0' in the binary array, 
        # we map it to an actual row in dataset_T vs. a row in dataset_R, etc.
        bin_seq_pp = binary_sequences_pp.astype(int).copy()
        neigh_idx_pp = neighborhood_idx_pp.astype(int).copy()
        neigh_idx_pp[bin_seq_pp == 0] += dataset_T.shape[0]

        # For the row(s) with the highest Upsilon, collect neighbors 
        Neig_indexes_temp = neigh_idx_pp[index_max, :K_M].astype(int)
        unique_elements   = collect_unique_until_greater(Neig_indexes_temp, dataset_T.shape[0])

        # Temporarily mark them as removed
        Upsilon_i_T_wrt_R_temp                  = Upsilon_i_T_wrt_R.copy()
        Upsilon_i_T_wrt_R_temp[unique_elements] = -1
        Upsilon_i_temp                          = Upsilon_i_T_wrt_R_temp[list_temp].copy()
        results_container                       = {}
        unique_elements_temp_dict               = {}


        print(f"Max Upsilon remained: {Upsilon_i_temp.max()}")
        if unique_elements.any():
            unique_elements_temp_dict[key_thresh] = unique_elements
        else:
            unique_elements_temp_dict[key_thresh] = list(list_temp[index_max])
        
        while Upsilon_i_temp.max() >= Upsilon_star_plus: 

            indices_updated = [
                i for i, row in enumerate(neigh_idx_pp) if row[0] not in unique_elements
            ]

            # Exclude elements from unique_elements and keep first K_M neighbors
            NEW_neigh = exclude_and_take_first_k(
                neigh_idx_pp[indices_updated, :], unique_elements, K_M, -1
            )
            NEW_binary_sequence = (NEW_neigh < dataset_T.shape[0]).astype(int)

            # Recompute the p-values for these updated neighbor sets
            KSTAR_RANGE = range(20, K_M)
            stats_local = calculate_p_values(NEW_binary_sequence, kstar_range=KSTAR_RANGE,p=p )

            # Update only for the sub-list
            Upsilon_i_temp[indices_updated] = stats_local['Upsilon_i_plus']

            if Upsilon_i_temp.max() < Upsilon_star_plus:
                break
            
            else:
                index_max = np.where(Upsilon_i_temp == Upsilon_i_temp.max())[0]
                Neig_indexes_temp = neigh_idx_pp[index_max].astype(int)

                unique_elements_temp = collect_unique_until_greater(
                    Neig_indexes_temp, dataset_T.shape[0]
                )
                if len(unique_elements_temp) == 0:
                    # Fallback: if no neighbors are found, pick the single row 
                    # with the highest Upsilon
                    unique_elements_temp = [Neig_indexes_temp[0][0]]
                    unique_elements = list(set(unique_elements) | set(unique_elements_temp))

                    # Mark them as removed
                    to_remove_idx = [
                        i for i, row in enumerate(neigh_idx_pp)
                        if row[0] in [Neig_indexes_temp[0][0]]
                    ]
                    Upsilon_i_temp[to_remove_idx] = -1
                    Upsilon_i_temp[
                        [i for i, row in enumerate(neigh_idx_pp) if row[0] in unique_elements]
                    ] = -1
                else:
                    unique_elements = list(set(unique_elements) | set(unique_elements_temp))
                    Upsilon_i_temp[
                        [i for i, row in enumerate(neigh_idx_pp) if row[0] in unique_elements]
                    ] = -1

            print(f"Max Upsilon remained: {Upsilon_i_temp.max()}")
            key_thresh = Upsilon_i_temp.max()
            unique_elements_temp_dict[key_thresh] = unique_elements_temp

        return unique_elements_temp_dict


    else:
        print("Warning! The smallest value of Upsilon_star_plus is greater than or equal to the smallest key in unique_elements_dict!")
        return unique_elements_dict



# def compute_anomalous_region(reference_data, data_with_anomaly, Upsilon_i_minus, Upsilon_star_minus, Upsilon_i_plus, Upsilon_star_plus,EXCESS_OVER, EXCESS_UNDER, smoothing,NUMBER_CORES, PARTITION_SIZE):
#     import From_data_to_binary
#     bin_nn_smoothing, NN_smoothing = From_data_to_binary.create_binary_array_cdist_post(data_with_anomaly, reference_data, smoothing+2, NUMBER_CORES,None,PARTITION_SIZE)
    
#     bin_nn_smoothing = bin_nn_smoothing.astype(int)
#     NN_smoothing = NN_smoothing.astype(int)

#     EXCESS_UNDER_LOC = [x + data_with_anomaly.shape[0] for x in EXCESS_UNDER]
#     for jj in range(smoothing):
#         NN_smoothing[bin_nn_smoothing[:,jj+1]==0,jj+1] = NN_smoothing[bin_nn_smoothing[:,jj+1]==0,1] + data_with_anomaly.shape[0]

#     mask_under = np.isin(NN_smoothing[:, 1:], EXCESS_UNDER_LOC)
#     mask_over  = np.isin(NN_smoothing[:, 1:], EXCESS_OVER)
    
#     rows_with_match_under = np.any(mask_under, axis=1)
#     rows_with_match_over  = np.any(mask_over, axis=1)
    
#     # Convert lists to sets
#     set1_under = set(np.where(rows_with_match_under)[0])
#     set2_under = set(np.where(Upsilon_i_minus>Upsilon_star_minus)[0])

#     # Find intersection
#     REGION_UNDER = list(set1_under.intersection(set2_under))
    
#     # Convert lists to sets
#     set1_over = set(np.where(rows_with_match_over)[0])
#     set2_over = set(np.where(Upsilon_i_plus>Upsilon_star_plus)[0])

#     # Find intersection
#     REGION_OVER = list(set1_over.intersection(set2_over))
   
#     return REGION_UNDER, REGION_OVER

def pval_post_equalization(
    test_data, 
    reference_data, 
    subset_indices,
    K_M,
    NUMBER_CORES,
    PARTITION_SIZE
    ):
    
    import From_data_to_binary_post
    binary_sequences_pp, neighborhood_idx_pp = From_data_to_binary_post.create_binary_array_cdist_post_subset(
        test_data[subset_indices,:],
        reference_data,
        range(len(subset_indices)),
        num_neighbors=K_M ,
        num_cores=NUMBER_CORES,
        validation=None,
        partition_size=PARTITION_SIZE
    )    
    p = len(test_data) / (len(test_data) + len(reference_data))
    KSTAR_RANGE = range(20,K_M)
    
    stats_local = calculate_p_values(binary_sequences_pp, kstar_range=KSTAR_RANGE,p=p )

            # Update only for the sub-list
    
    
    return stats_local['Upsilon_i_plus']









def get_indicies(thresh,res_new):
    if thresh > max(res_new.keys()):
        return thresh, []
    keys = np.array(list(res_new.keys()))
    keys_new = [key for key in keys if key >= thresh]
    # concatenate all the ites for elements of the dict with keys_new
    inds = np.concatenate([res_new[key] for key in keys_new])
    return inds


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Soar(reference_data, test_data, result_dict_in={}, K_M=1000, critical_quantiles=None, num_cores=1, validation=None, partition_size=100,smoothing=3):
    print("Eagle...Soar!")
    print(r"""
                               /T /I
                              / |/ | .-~/
                          T\ Y  I  |/  /  _
         /T               | \I  |  I  Y.-~/
        I l   /I       T\ |  |  l  |  T  /
     T\ |  \ Y l  /T   | \I  l   \ `  l Y
 __  | \l   \l  \I l __l  l   \   `  _. |
 \ ~-l  `\   `\  \  \ ~\  \   `. .-~   |
  \   ~-. "-.  `  \  ^._ ^. "-.  /  \   |
.--~-._  ~-  `  _  ~-_.-"-." ._ /._ ." ./
 &gt;--.  ~-.   ._  ~&gt;-"    "\   7   7   ]
^.___~"--._    ~-{  .-~ .  `\ Y . /    |
 &lt;__ ~"-.  ~       /_/   \   \I  Y   : |
   ^-.__           ~(_/   \   &gt;._:   | l______
       ^--.,___.-~"  /_/   !  `-.~"--l_ /     ~"-.
              (_/ .  ~(   /'     "~"--,Y   -=b-. _)
               (_/ .  \  :           / l      c"~o \
                \ /    `.    .     .^   \_.-~"~--.  )
                 (_/ .   `  /     /       !       )/
                  / / _.   '.   .':      /        '
                  ~(_/ .   /    _  `  .-&lt;_
                    /_/ . ' .-~" `.  / \  \          ,z=.
                    ~( /   '  :   | K   "-.~-.______//
                      "-,.    l   I/ \_    __{---&gt;._(==.
                       //(     \  &lt;    ~"~"     //
                      /' /\     \  \     ,v=.  ((
                    .^. / /\     "  }__ //===-  `
                   / / ' '  "-.,__ {---(==-
                 .^ '       :  T  ~"   ll      
                / .  .  . : | :!        \
               (_/  /   | | j-"          ~^
                 ~-&lt;_(_.^-~"
""", end='')
    
    result_dict = result_dict_in.copy()
    # Determin lists of upsilon stars for thresholds
    if critical_quantiles is not None:
        print("Critical quantiles detected. Computing null distribution!")
        KSTAR_RANGE                                    = range(20, K_M) # Range of kstar values to consider
        num_sequences                                  = 500000 # Hardcoded for good stats
        p                                              = len(test_data) / (len(test_data) + len(reference_data))
        binary_sequences                               = np.random.binomial(n=1, p=p, size=(num_sequences, K_M))
        stats_null                                     = calculate_p_values(binary_sequences, kstar_range=KSTAR_RANGE, p=p, validation=validation)
        del(binary_sequences)
        Upsilon_i_plus_null  = stats_null['Upsilon_i_plus']
        Upsilon_i_minus_null = stats_null['Upsilon_i_minus']

        # Critical thresholds as defined in the article
        Upsilon_star_plus  = np.quantile(Upsilon_i_plus_null,  critical_quantiles)
        Upsilon_star_minus = np.quantile(Upsilon_i_minus_null, critical_quantiles)
        result_dict['Upsilon_star_plus']   = Upsilon_star_plus
        result_dict['Upsilon_star_minus']  = Upsilon_star_minus
        result_dict['critical_quantiles']  = critical_quantiles
        result_dict['Upsilon_i_plus_null'] = Upsilon_i_plus_null
        result_dict['Upsilon_i_minus_null']= Upsilon_i_minus_null
    else:
        result_dict['Upsilon_star_plus']   = None
        result_dict['Upsilon_star_minus']  = None
        result_dict['critical_quantiles']  = None
        result_dict['Upsilon_i_plus_null'] = None
        result_dict['Upsilon_i_minus_null']= None


    if not result_dict.get('stats') or not result_dict.get('stats_reverse'):
        # Initialise res dict if EE not been run before. 
        # I.e obtain Upsilon+ and Upsilon- lists 
        KSTAR_RANGE                                    = range(20, K_M) # Range of kstar values to consider
        NUMBER_CORES                                   = num_cores  
        PARTITION_SIZE                                 = partition_size
        p                                              = len(test_data) / (len(test_data) + len(reference_data))
        #%% Compute overdensities & inject validation
        print("Compute overdensities")
        binary_sequences                               = From_data_to_binary.create_binary_array_cdist(test_data, reference_data, num_neighbors=K_M, num_cores=NUMBER_CORES, validation=validation,partition_size=PARTITION_SIZE)
        stats                                          = calculate_p_values(binary_sequences, kstar_range=KSTAR_RANGE, p=p, validation=validation,verbose=True)
        del(binary_sequences)
        #%% compute underdensities & IV
        print("Compute underdensities")
        binary_sequences_reverse                       = From_data_to_binary.create_binary_array_cdist(reference_data, test_data, num_neighbors=K_M, num_cores=NUMBER_CORES, validation=validation,partition_size=PARTITION_SIZE)
        stats_reverse                                  = calculate_p_values(binary_sequences_reverse, kstar_range=KSTAR_RANGE, p=p, validation=validation,verbose=True)
        del(binary_sequences_reverse)
        result_dict['stats']                          = stats
        result_dict['stats_reverse']                  = stats_reverse
        result_dict['unique_elements_overdensities']  = {}
        result_dict['unique_elements_underdensities'] = {}
        result_dict['overdensities']                  = {}
        result_dict['underdensities']                 = {}
#%%    equalize the overdensities
    if critical_quantiles:
        # Iterative equilisation (halo removal) 
        # Check if list of quantiles exists or can be reused
        threshP = min(Upsilon_star_plus)
        threshM  = min(Upsilon_star_minus)
        unique_keys_o = list(result_dict.get('unique_elements_overdensities', {}).keys())
        if (not result_dict.get('overdensities')) or (unique_keys_o and threshP < unique_keys_o[-1]):
        # if not result_dict.get('overdensities') or threshP < min(result_dict['critical_quantiles']) :
            print("Computing unique elements for quantile/s.")
            result_dict['unique_elements_overdensities']                  = iterative_equalization(
            result_dict['unique_elements_overdensities'],
            test_data, 
            reference_data, 
            result_dict['stats']['Upsilon_i_plus'], 
            threshP,
            K_M,
            num_cores,
            partition_size
            )

            
       #%% equalize the underdensities
        unique_keys_u = list(result_dict.get('unique_elements_underdensities', {}).keys())
        if (not result_dict.get('underdensities')) or (unique_keys_u and threshM < unique_keys_u[-1]): 
            result_dict['unique_elements_underdensities']                  = iterative_equalization(
                result_dict['unique_elements_underdensities'],
                reference_data, 
                test_data, 
                result_dict['stats_reverse']['Upsilon_i_plus'], 
                threshM,
                K_M,
                num_cores,
                partition_size
            )        
        else:
            print("Reusing unique elements for quantile/s.")
#%%          
        # Save the useful disctionaries with all indices for anomolous regions for each quantile
        result_dict['overdensities']                  = {thresh:get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_overdensities']) for thresh in critical_quantiles}
        result_dict['underdensities']                 = {thresh:get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_underdensities']) for thresh in critical_quantiles}            


    # #%% get back
    # REGION_UNDER, REGION_OVER = compute_anomalous_region(reference_data,
    #                                                                 test_data,
    #                                                                 result_dict['stats']['Upsilon_i_minus'],
    #                                                                 result_dict['Upsilon_star_minus'],
    #                                                                 result_dict['stats']['Upsilon_i_plus'],
    #                                                                 result_dict['Upsilon_star_plus'],
    #                                                                 unique_elements_overdensities,
    #                                                                 unique_elements_underdensities,
    #                                                                 smoothing,
    #                                                                 NUMBER_CORES,
    #                                                                 PARTITION_SIZE)

    # result_dict['REGION_UNDER'] = REGION_UNDER
    # result_dict['REGION_OVER'] = REGION_OVER

#%%


    # Print shapes of all keys of result_dict in a nice table
    print("\nShapes of result_dict keys:")
    print("{:<30} {:<15}".format("Key", "Shape"))
    print("-" * 45)
    for key, value in result_dict.items():
        if isinstance(value, np.ndarray):
            shape = value.shape
        elif isinstance(value, list):
            shape = (len(value),)
        elif isinstance(value, dict):
            print("{:<30} {:<15}".format(key, "dict"))
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    sub_shape = sub_value.shape
                elif isinstance(sub_value, list):
                    sub_shape = (len(sub_value),)
                else:
                    sub_shape = "N/A"
                print("  {:<28} {:<15}".format(sub_key, str(sub_shape)))
            continue
        else:
            shape = "N/A"
        print("{:<30} {:<15}".format(key, str(shape)))

    return result_dict





############################################################################################################
# Partitianing and BPR estimation after iterative equalisation 

def cluster(data,K_M,Z=1.65):
    # Adjust maxk based on the number of samples
    data.compute_distances(maxk=K_M)
    data.compute_id_2NN()
    data.compute_density_kstarNN()
    data.compute_clustering_ADP(Z=Z, halo=False)
    return data 

def partitian_function(reference_data,test_data,result_dict,Upsilon_star_plus, Upsilon_star_minus,K_M,Z=1.65):
    # For all points in the dataset, we will now partition them into groups with DPA clustering
    Upsilon_i_plus  = result_dict['stats']['Upsilon_i_plus']
    Upsilon_i_Val_plus  = result_dict['stats']['Upsilon_i_Val_plus']
    
    
    Upsilon_i_plus_rev = result_dict['stats_reverse']['Upsilon_i_plus']
    Upsilon_i_Val_plus_rev = result_dict['stats_reverse']['Upsilon_i_Val_plus']
    
    
    X_plus          = test_data[Upsilon_i_plus > Upsilon_star_plus]
    X_plus_val      = reference_data[Upsilon_i_Val_plus > Upsilon_star_plus]
    
    X_minus         = reference_data[Upsilon_i_plus_rev > Upsilon_star_minus]
    X_minus_val     = test_data[Upsilon_i_Val_plus_rev  > Upsilon_star_minus]
    
###############################################################################    
    UP = np.concatenate( (Upsilon_i_plus,Upsilon_i_Val_plus) )
    UM = np.concatenate( (Upsilon_i_Val_plus_rev,Upsilon_i_plus_rev) )
    
    XP = np.concatenate( (X_plus,X_plus_val) )
    XM = np.concatenate( (X_minus_val,X_minus) )
    
    indx_plus       = np.where(UP > Upsilon_star_plus)[0]
    if len(indx_plus)>3:
        data_plus       = Data(XP, verbose=True)
        data_plus       = cluster(data_plus,min(K_M,X_plus.shape[0]-1),Z=Z)
        clusters_plus   = [indx_plus[data_plus.cluster_assignment == i] for i in range(len(data_plus.cluster_centers))]
    else:
        clusters_plus = [ list(indx_plus)] 

    indx_minus      = np.where(UM > Upsilon_star_minus)[0]
    if len(indx_minus)>3:
        data_minus      = Data(XM, verbose=True)
        data_minus      = cluster(data_minus,min(K_M,X_minus.shape[0]-1),Z=Z)
        clusters_minus  = [indx_minus[data_minus.cluster_assignment == i] for i in range(len(data_minus.cluster_centers))]
    else:
        clusters_minus = [ list(indx_minus)] 

    return clusters_plus,clusters_minus

# def IV_IE(clusters,result_dict,thresh):
#     clusters_plus,clusters_minus = clusters
#     overdensity_indicies_plus = get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_overdensities'])
#     print(overdensity_indicies_plus)

#     overdensity_indicies_minus = get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_underdensities'])
#     IV_IE_list_plus = []
#     IV_IE_list_minus = []
#     ii = 0
#     print("Number of clusters: ", len(clusters_plus))   
#     for cluster in clusters_plus:
#         ii+=1
#         print("Cluster number ", ii)
#         # Get intersection of overdensity and cluster
#         intersection          = list(set(overdensity_indicies_plus).intersection(set(cluster)))
#         if ii==2:
#             print(intersection)
#             print("----------------------------")
#             print(cluster)
        
#         minimum_upsilion_plus = np.min(result_dict['stats']['Upsilon_i_plus'][intersection])
#         idx_gt_min_plus       = [x for x in cluster if result_dict['stats']['Upsilon_i_plus'][x] >= minimum_upsilion_plus]
#         IV_IE_list_plus.append(idx_gt_min_plus)
#     for cluster in clusters_minus:
#         # Get intersection of underdensity and cluster
#         intersection           = list(set(overdensity_indicies_minus).intersection(set(cluster)))
#         minimum_upsilion_minus = np.min(result_dict['stats_reverse']['Upsilon_i_plus'][intersection])
#         idx_gt_min_minus       = [x for x in cluster if result_dict['stats_reverse']['Upsilon_i_plus'][x] >= minimum_upsilion_minus]
#         IV_IE_list_minus.append(idx_gt_min_minus)
#     return IV_IE_list_plus,IV_IE_list_minus

def IV_IE_get_dict(clusters,result_dict,thresh, data_with_anomaly, reference_data):
    clusters_plus,clusters_minus = clusters
    IV_IE_dict = {
    "OVER_clusters": {
        i: {"IE_extra": None, "From_test": None, "From_ref": None}
        for i in range(len(clusters_plus))
    },
    "UNDER_clusters": {
        i: {"IE_extra": None, "From_test": None, "From_ref": None}
        for i in range(len(clusters_minus))
    }
    }
    clusters_plus,clusters_minus = clusters
    if result_dict['unique_elements_overdensities']:
        overdensity_indicies_plus = get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_overdensities'])
    else:
        overdensity_indicies_plus = []
    if result_dict['unique_elements_underdensities']:
        overdensity_indicies_minus = get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_underdensities'])
    else:
        overdensity_indicies_minus = []
    # IV_IE_list_plus = []
    # IV_IE_list_minus = []
    ii = 0
    print("Number of clusters: ", len(clusters_plus))
    for cluster in clusters_plus:
        cluster   = np.array( cluster )
        cluster_r = cluster[cluster>=result_dict['stats']['Upsilon_i_Val_plus'].shape[0]] - result_dict['stats']['Upsilon_i_Val_plus'].shape[0]
        cluster_t = cluster[cluster<result_dict['stats']['Upsilon_i_plus'].shape[0]]
        # # Create the scatter plot and capture the scatter object
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(
        #         data_with_anomaly[cluster_t, 0],
        #         data_with_anomaly[cluster_t, 1],
        #         data_with_anomaly[cluster_t, 2],
        #         alpha=1,
        #         s=5
        #     )
        # ax.scatter(
        #         reference_data[cluster_r, 0],
        #         reference_data[cluster_r, 1],
        #         reference_data[cluster_r, 2],
        #         alpha=1,
        #         s=5
        #     )
        # ax.set_xlim(-100,100)
        # ax.set_ylim(-100,100)
        # ax.set_zlim(-100,100)
        # plt.title("debug ")
        # plt.show()
        ii+=1
        print("Cluster number ", ii)
        # Get intersection of overdensity and cluster
        intersection          = list(set(overdensity_indicies_plus).intersection(set(cluster_t)))
        if intersection:
        
            minimum_upsilion_plus = np.min(result_dict['stats']['Upsilon_i_plus'][intersection])
            IV_IE_dict['OVER_clusters'][ii-1]['IE_extra'] = intersection
            IV_IE_dict['OVER_clusters'][ii-1]['From_test'] = [x for x in cluster_t if result_dict['stats']['Upsilon_i_plus'][x] >= minimum_upsilion_plus]
            IV_IE_dict['OVER_clusters'][ii-1]['From_ref'] = [x for x in cluster_r if result_dict['stats']['Upsilon_i_Val_plus'][x] >= minimum_upsilion_plus]

    ############################################################################
    ii = 0
    print("Number of clusters: ", len(clusters_minus))
    for cluster in clusters_minus:
        cluster   = np.array( cluster )
        cluster_r = cluster[cluster>=result_dict['stats_reverse']['Upsilon_i_Val_plus'].shape[0]] - result_dict['stats']['Upsilon_i_Val_plus'].shape[0]
        cluster_t = cluster[cluster<result_dict['stats_reverse']['Upsilon_i_plus'].shape[0]]
        # # Create the scatter plot and capture the scatter object
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(
        #         data_with_anomaly[cluster_t, 0],
        #         data_with_anomaly[cluster_t, 1],
        #         data_with_anomaly[cluster_t, 2],
        #         alpha=1,
        #         s=5
        #     )
        # ax.scatter(
        #         reference_data[cluster_r, 0],
        #         reference_data[cluster_r, 1],
        #         reference_data[cluster_r, 2],
        #         alpha=1,
        #         s=5
        #     )
        # ax.set_xlim(-100,100)
        # ax.set_ylim(-100,100)
        # ax.set_zlim(-100,100)
        # plt.title("debug ")
        # plt.show()
        ii+=1
        # Get intersection of underdensity and cluster
        intersection           = list(set(overdensity_indicies_minus).intersection(set(cluster_r)))
        if intersection:
        
            minimum_upsilion_minus = np.min(result_dict['stats_reverse']['Upsilon_i_plus'][intersection])
            # idx_gt_min_minus       = [x for x in cluster if result_dict['stats_reverse']['Upsilon_i_plus'][x] >= minimum_upsilion_minus]
            # IV_IE_list_minus.append(idx_gt_min_minus)
            IV_IE_dict['UNDER_clusters'][ii-1]['IE_extra'] = intersection
            IV_IE_dict['UNDER_clusters'][ii-1]['From_test'] = [x for x in cluster_t if result_dict['stats_reverse']['Upsilon_i_Val_plus'][x] >= minimum_upsilion_minus]
            IV_IE_dict['UNDER_clusters'][ii-1]['From_ref'] = [x for x in cluster_r if result_dict['stats_reverse']['Upsilon_i_plus'][x] >= minimum_upsilion_minus]
    return IV_IE_dict#IV_IE_list_plus,IV_IE_list_minus










# def IV_IE_get_dict(clusters,result_dict,thresh, data_with_anomaly, reference_data):
    
#     clusters_plus,clusters_minus = clusters
    
#     IV_IE_dict = {
#     "OVER_clusters": {
#         i: {"IE_extra": None, "From_test": None, "From_ref": None}
#         for i in range(len(clusters_plus))
#     },
#     "UNDER_clusters": {
#         i: {"IE_extra": None, "From_test": None, "From_ref": None}
#         for i in range(len(clusters_minus))
#     }
#     }
    
    
    
#     clusters_plus,clusters_minus = clusters
#     overdensity_indicies_plus = get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_overdensities'])
#     print(overdensity_indicies_plus)

#     overdensity_indicies_minus = get_indicies(np.quantile(result_dict['Upsilon_i_plus_null'],  thresh),result_dict['unique_elements_underdensities'])
    
#     # IV_IE_list_plus = []
#     # IV_IE_list_minus = []
#     ii = 0
#     print("Number of clusters: ", len(clusters_plus))   
#     for cluster in clusters_plus:
#         cluster   = np.array( cluster )
#         cluster_r = cluster[cluster>=result_dict['stats']['Upsilon_i_Val_plus'].shape[0]] - result_dict['stats']['Upsilon_i_Val_plus'].shape[0]
#         cluster_t = cluster[cluster<result_dict['stats']['Upsilon_i_plus'].shape[0]]
        
#         # # Create the scatter plot and capture the scatter object
#         # fig = plt.figure(figsize=(10, 8))
#         # ax = fig.add_subplot(111, projection='3d')
#         # ax.scatter(
#         #         data_with_anomaly[cluster_t, 0],
#         #         data_with_anomaly[cluster_t, 1],
#         #         data_with_anomaly[cluster_t, 2],
#         #         alpha=1,
#         #         s=5
#         #     )
#         # ax.scatter(
#         #         reference_data[cluster_r, 0],
#         #         reference_data[cluster_r, 1],
#         #         reference_data[cluster_r, 2],
#         #         alpha=1,
#         #         s=5
#         #     )
#         # ax.set_xlim(-100,100)
#         # ax.set_ylim(-100,100)
#         # ax.set_zlim(-100,100)

#         # plt.title("debug ")
#         # plt.show()
        
#         ii+=1
#         print("Cluster number ", ii)
#         # Get intersection of overdensity and cluster
#         intersection          = list(set(overdensity_indicies_plus).intersection(set(cluster_t)))
#         if ii==2:
#             print(intersection)
#             print("----------------------------")
#             print(cluster)
        
#         minimum_upsilion_plus = np.min(result_dict['stats']['Upsilon_i_plus'][intersection])
#         # idx_gt_min_plus       = [x for x in cluster if result_dict['stats']['Upsilon_i_plus'][x] >= minimum_upsilion_plus]
        
#         # IV_IE_list_plus.append(idx_gt_min_plus)
        
#         IV_IE_dict['OVER_clusters'][ii-1]['IE_extra'] = intersection
#         IV_IE_dict['OVER_clusters'][ii-1]['From_test'] = [x for x in cluster_t if result_dict['stats']['Upsilon_i_plus'][x] >= minimum_upsilion_plus]
#         IV_IE_dict['OVER_clusters'][ii-1]['From_ref'] = [x for x in cluster_r if result_dict['stats']['Upsilon_i_Val_plus'][x] >= minimum_upsilion_plus]
        
#     ############################################################################   
#     ii = 0
#     print("Number of clusters: ", len(clusters_minus))      
#     for cluster in clusters_minus:
#         cluster   = np.array( cluster )
#         cluster_r = cluster[cluster>=result_dict['stats_reverse']['Upsilon_i_Val_plus'].shape[0]] - result_dict['stats']['Upsilon_i_Val_plus'].shape[0]
#         cluster_t = cluster[cluster<result_dict['stats_reverse']['Upsilon_i_plus'].shape[0]]
        
#         # # Create the scatter plot and capture the scatter object
#         # fig = plt.figure(figsize=(10, 8))
#         # ax = fig.add_subplot(111, projection='3d')
#         # ax.scatter(
#         #         data_with_anomaly[cluster_t, 0],
#         #         data_with_anomaly[cluster_t, 1],
#         #         data_with_anomaly[cluster_t, 2],
#         #         alpha=1,
#         #         s=5
#         #     )
#         # ax.scatter(
#         #         reference_data[cluster_r, 0],
#         #         reference_data[cluster_r, 1],
#         #         reference_data[cluster_r, 2],
#         #         alpha=1,
#         #         s=5
#         #     )
#         # ax.set_xlim(-100,100)
#         # ax.set_ylim(-100,100)
#         # ax.set_zlim(-100,100)

#         # plt.title("debug ")
#         # plt.show()
        
#         ii+=1
        
        
#         # Get intersection of underdensity and cluster
#         intersection           = list(set(overdensity_indicies_minus).intersection(set(cluster_r)))
#         minimum_upsilion_minus = np.min(result_dict['stats_reverse']['Upsilon_i_plus'][intersection])
#         # idx_gt_min_minus       = [x for x in cluster if result_dict['stats_reverse']['Upsilon_i_plus'][x] >= minimum_upsilion_minus]
#         # IV_IE_list_minus.append(idx_gt_min_minus)
        
        
        

#         IV_IE_dict['UNDER_clusters'][ii-1]['IE_extra'] = intersection
#         IV_IE_dict['UNDER_clusters'][ii-1]['From_test'] = [x for x in cluster_t if result_dict['stats_reverse']['Upsilon_i_Val_plus'][x] >= minimum_upsilion_minus]
#         IV_IE_dict['UNDER_clusters'][ii-1]['From_ref'] = [x for x in cluster_r if result_dict['stats_reverse']['Upsilon_i_plus'][x] >= minimum_upsilion_minus]
        
        
#     return IV_IE_dict#IV_IE_list_plus,IV_IE_list_minus

























