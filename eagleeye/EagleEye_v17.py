#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Fri 7 15:38:20 2025

@author: sspringe, AndreScaffidi
"""

import numpy as np
from scipy.stats import binom
from IPython.display import display, Math



# ----------------------------------------------------------------------
# 1) PValueCalculator Class
# ----------------------------------------------------------------------


class PValueCalculator:
    """
    Computes for each row in a binary array the negative-log p-values (plus tail)
    for various subsequence lengths k, and finds the row-wise maximum.
    """
    def __init__(self, binary_sequence, kstar_range, p=0.5, pvals_dict=None, verbose=False):
        """
        Parameters
        ----------
        binary_sequence : np.ndarray of shape (N, M)
            Binary (0/1) data array.
        kstar_range : iterable of ints
            List (or range) of subsequence lengths k for which to compute p-values.
        p : float, optional
            Probability parameter for the Binomial distribution (default=0.5).
        pvals_dict : dict or None, optional
            Optional precomputed dict of plus-tail p-values.
        verbose : bool, optional
            If True, prints progress information.
        """
        self.binary_sequence = binary_sequence
        self.kstar_range = list(kstar_range)
        self.p = p
        self.verbose = verbose

        # Precompute cumulative sums along rows for fast summation of first k bits.
        self.binary_cumsum = np.c_[
            np.zeros((binary_sequence.shape[0], 1), dtype=int),
            np.cumsum(binary_sequence, axis=1)
        ]

        # Build (or reuse) the dictionary of negative-log plus-tail p-values.
        if pvals_dict is not None:
            self.pvals_dict = pvals_dict
        else:
            self.pvals_dict = self.build_pvals_dict()

        # For each k in kstar_range, compute an array of negative-log plus-tail p-values.
        self.pval_array_dict = {}
        for k in self.kstar_range:
            self.pval_array_dict[k] = self.compute_neglog_array(k)
            if self.verbose and (k % 100 == 0):
                print(f"Computed p-values for k = {k}")

        # Compute row-wise maximum plus-tail p-values and the corresponding k values.
        self.compute_rowwise_maxima()

    def build_pvals_dict(self):
        """
        Build a dictionary that, for each k in kstar_range and each possible sum (0..k),
        stores the negative-log plus-tail p-value.

        Returns
        -------
        dict
            pvals_dict[k] is a NumPy array of shape (k+1,) containing the negative-log plus-tail
            p-values for sums 0, 1, ..., k.
        """
        pvals_dict = {}
        for k in self.kstar_range:
            x = np.arange(k + 1)
            pmf = binom.pmf(x, k, self.p)
            # Compute the cumulative distribution for the plus tail.
            cdf_plus = np.cumsum(pmf[::-1])[::-1]
            neg_log_plus = -np.log(cdf_plus)
            pvals_dict[k] = neg_log_plus
        return pvals_dict

    def compute_neglog_array(self, kstar):
        """
        For each row, sum up the first kstar binaries, then look up the negative-log plus-tail
        p-value using vectorized indexing.

        Returns
        -------
        np.ndarray
            An array of shape (N,) containing the negative-log plus-tail p-values for each row.
        """
        stat_sum = self.binary_cumsum[:, kstar]
        return self.pvals_dict[kstar][stat_sum]

    def compute_rowwise_maxima(self):
        """
        Compute the row-wise maximum negative-log plus-tail p-value and record the k value
        (kstar) that attains this maximum.

        Attributes
        ----------
        self.min_pval_plus : np.ndarray of shape (N,)
            Maximum -log(p_plus) among all k for each row.
        self.kstar_min_pval_plus : np.ndarray of shape (N,)
            The k value that attains the maximum -log(p_plus) per row.
        """
        # Stack plus-tail p-value arrays across all k values.
        plus_vals = np.column_stack([self.pval_array_dict[k] for k in self.kstar_range])
        self.min_pval_plus = np.max(plus_vals, axis=1)
        kstar_array = np.array(self.kstar_range)
        self.kstar_min_pval_plus = kstar_array[np.argmax(plus_vals, axis=1)]
#%%
# ----------------------------------------------------------------------
# 2) Iterative density equalization 
# ----------------------------------------------------------------------


def IDE_step_optimized(Knn_model, X, Y, putative_indices, banned_set, K_M, p, 
                         Upsilon_i, Upsilon_star_plus, neighbors_dict, nX):
    """
    Optimized version of IDE_step that vectorizes filtering and batches KNN recomputation.
    """
    # Retrieve precomputed new_indices for the current putative_indices.
    new_indices = np.array([neighbors_dict[i] for i in putative_indices])
    # nX = X.shape[0]
    
    # Determine overdensity type once.
    is_overdensity = new_indices[0, 0] >= nX
    if is_overdensity:
        banned_set = {x + nX for x in banned_set}
    
    # Convert banned_set to a NumPy array once.
    banned_array = np.array(list(banned_set))
    
    # Vectorized filtering: Compute a mask for valid (non-banned) elements in each row.
    mask = ~np.isin(new_indices, banned_array)
    valid_counts = mask.sum(axis=1)
    
    # Initialize the output filtered indices array.
    num_points = new_indices.shape[0]
    filtered_indices = np.empty((num_points, K_M), dtype=int)
    
    # For rows that already have enough valid entries, fill the filtered_indices.
    rows_ok = np.where(valid_counts >= K_M)[0]
    if rows_ok.size > 0:
        # Use list comprehension with pre-filtered rows.
        valid_rows = [new_indices[i, :][mask[i]][:K_M] for i in rows_ok]
        for idx, row in zip(rows_ok, valid_rows):
            filtered_indices[idx, :] = row
    
    # Identify rows that need KNN re-computation.
    rows_to_update = np.where(valid_counts < K_M)[0]
    if rows_to_update.size > 0:
        # Batch update: Get corresponding putative indices for these rows.
        indices_to_update = [putative_indices[i] for i in rows_to_update]
        
        # Set the number of neighbors for re-computation.
        Knn_model.n_neighbors = int((X.shape[0] + Y.shape[0]) * 0.05)
        # Batch KNN query for all rows that need update.
        new_rows = Knn_model.kneighbors(Y[indices_to_update, :])[1]  # shape (num_updates, n_neighbors)
        
        # Process each updated row.
        for idx, row_idx in enumerate(rows_to_update):
            new_row = new_rows[idx]
            # Update the neighbors_dict.
            neighbors_dict[putative_indices[row_idx]] = new_row
            # Apply the banned filter.
            row_mask = ~np.isin(new_row, banned_array)
            row_filtered = new_row[row_mask]
            # Ensure we have K_M elements (pad if needed).
            if row_filtered.size < K_M:
                # For simplicity, we pad by repeating the last valid element (or zero if none valid).
                pad_value = row_filtered[-1] if row_filtered.size > 0 else 0
                padded = np.pad(row_filtered, (0, K_M - row_filtered.size), 
                                  mode='constant', constant_values=pad_value)
                filtered_indices[row_idx, :] = padded
            else:
                filtered_indices[row_idx, :] = row_filtered[:K_M]
    
    # Compute the binary location matrix.
    if is_overdensity:
        binary_loc = (filtered_indices > nX).astype(int)
    else:
        binary_loc = (~(filtered_indices > nX)).astype(int)
    
    # Use the existing PValueCalculator to compute p-values.
    KSTAR_RANGE = range(20, K_M)
    p_val_info = PValueCalculator(binary_loc, KSTAR_RANGE, p=p)
    
    # Identify the index of the maximum p-value.
    max_p_val = p_val_info.min_pval_plus.max()
    index_max = np.where(p_val_info.min_pval_plus == max_p_val)[0]
    
    # Collect unique elements based on the overdensity type.
    if is_overdensity:
        unique_elements = collect_unique_(filtered_indices[index_max, :], nX, True)
        unique_elements_l = [z - nX for z in unique_elements if Upsilon_i[z - nX] > Upsilon_star_plus]
    else:
        unique_elements = collect_unique_(filtered_indices[index_max, :], nX, False)
        unique_elements_l = [z for z in unique_elements if Upsilon_i[z] > Upsilon_star_plus]
    
    # Update putative indices based on p-value threshold.
    updated_putative = [elem for elem, stat in zip(putative_indices, p_val_info.min_pval_plus)
                        if stat > Upsilon_star_plus]
    
    return unique_elements_l, max_p_val, updated_putative
def IDE(Y_IDE, Y, X, Upsilon_i, Upsilon_star_plus, K_M, p, n_jobs, Knn_model, nX):
    """
    Iteratively equalize overdensities of Y with respect to X.
    """
    # Identify putative indices based on the anomaly score.
    putative_indices = np.where(Upsilon_i > Upsilon_star_plus)[0]

    # Build the banned set from already computed values.
    banned_set = set()
    for sublist in Y_IDE.values():
        banned_set.update(sublist)

    # Precompute KNN for all putative indices.
    if putative_indices.size > 0:
        precomputed_neighbors = Knn_model.kneighbors(Y[putative_indices, :])[1]
        neighbors_dict = {idx: row for idx, row in zip(putative_indices, precomputed_neighbors)}
    else:
        neighbors_dict = {}

    # Determine which putative indices are not yet processed.
    putative_indices_left = [x for x in putative_indices if x not in banned_set]

    while putative_indices_left:
        pruned_step, key_step, putative_indices_left = IDE_step_optimized(
            Knn_model, X, Y, putative_indices_left, banned_set, K_M, p, Upsilon_i, Upsilon_star_plus, neighbors_dict, nX
        )
        Y_IDE[key_step] = pruned_step
        for sublist in Y_IDE.values():
            banned_set.update(sublist)
        putative_indices_left = [x for x in putative_indices_left if x not in banned_set]
    return Y_IDE


def collect_unique_(array_2d: np.ndarray, threshold: int, direction: bool) -> np.ndarray:
    """
    For each row, collect elements until encountering a value > threshold.
    Return unique values of all collected elements across rows.
    """
    if array_2d.size == 0:
        return np.array([], dtype=array_2d.dtype)

    slices = []
    for row in array_2d:
        if direction:
            mask = row < threshold
        else:
            mask = row >= threshold
        idx = np.argmax(mask)
        
        if mask[idx]:
            # Found at least one element > threshold
            # Take elements up to idx-1
            if idx==1:
                slices.append(row[:idx])
            else:
                slices.append(row[:idx-1])
        else:
            # No elements in this row exceed threshold
            slices.append(row)

    if slices:
        all_collected = np.concatenate(slices)
        return np.unique(all_collected)
    else:
        return np.array([], dtype=array_2d.dtype)
    
def get_indicies(thresh,res_new):
    if thresh > max(res_new.keys()):
        return thresh, []
    keys = np.array(list(res_new.keys()))
    keys_new = [key for key in keys if key >= thresh]
    # concatenate all the ites for elements of the dict with keys_new
    inds = np.concatenate([res_new[key] for key in keys_new])
    return inds

#%%
# ----------------------------------------------------------------------
# 3) SOAR!!!
# ----------------------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Soar(X, Y, K_M=500, p_ext=1e-5, n_jobs=10, stats_null={}, result_dict_in={}):
         
         
    print("-----------------------------------------------------------------")        
    print("Eagle...Soar!")
#     print(r"""
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*+++*#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*.               =@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@-                       %@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@#                            +@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@%@@@%                                -@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@%                                    #@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@:                                       @@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@                                          %@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@*                                            +@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@:                                              =@@@@@@@@@@@@@@
# @@@@@@@@@@@@@%.                                      #.        *@@@@@@@@@@@@@
# @@@@@@@@@@@@@.                                        %=-    .  @@@@@@@@@@@@@
# @@@@@@@@@@@@.                                          @#-:  --  @@@@@@@@@@@@
# @@@@@@@@@@@:                           :               :@%-- .-. %@@@@@@@@@@@
# @@@@@@@@@@*       .--+@@@%*.     .-      @-             %@*-- =%  @@@@@@@@@@@
# @@@@@@@@@@    :===--=*%%%@@@@@@*    @@+   @%=  :      +@@@@@@*-+@ @@@@@@@@@@@
# @@@@@@@@@        :-=%@@@@@@@@@@@@@@*  @@@* @@-= -    @@@@@@@@@%-@ @@@@@@@@@@@
# @@@@@@@@:      :===+@@@@@@@@@@@@@@@@@@@#@@@@@@-=-:  :@@-....:@@@@%@@@@@@@@@@@
# @@@@@@@%     .-=-%@@@@@@@@@@+:@@@@@@@@@@@@@@@@@-=-  @@:...... .@@@@@@@@@@@@@@
# @@@@@@@     --=@@@@@@@@@@@@:   @@@@@@@@@@@@@@@@%==  @@........   @@@@@@@@@@@@
# @@@@@@-.   --@@@@@@@@@@@@%#   :@@@%%@@@@@@@@@@@@--. @=.......     -@@@@@@@@@@
# @@@@@%   .-#@@@@@@@@@@@@%%+@ :@@@%#*@@@@@@@@@@@%=-..@:..:-....:     @@@@@@@@@
# @@@@@:  .-@@@@@######@@@%#=@@@@@@@+=@@@@@@@@@@@==-..@.:%@@%:...:.    #@@@@@@@
# @@@@@  .+@@@@ =---=###@@%#=%@@@@@@--@@@@@@@%%@@==:::%::%@@@-::::::    *@@@@@@
# @@@@- .#@@@ .=------##@@%#--@@@@@=-:@@%@@@@%%@-==:::%::@@@@::::::::.   %@@@@@
# @@@@..@@@+.      .=--#*@@#-:-%@@+-:#@@@@@@%#@===-:::+:-@@@%:::::::::.   @@@@@
# @@@%.@@@.        . ---#%@%#::---::*@%=@@@%##====::-:-:-@@%:::::::::::.   @@@@
# @@@.@@@.  .  .   ...=--=@@%#+::::#%@##@@=#=====-------*@#::-----------   -@@@
# @@@@@@. .............====@@@%%%%%@@+.@%==-===---------%=---------------   @@@
# @@@@@.................-===+@@@@@@% .@====.+----------------------------:  =@@
# @@@@....................-====++...*-=-. :===++=------------------------=   @@
# @@@+.............................-:.  *@@@@@@@@@@@%*-------------------=:  @@
# @@@...:........................... +%@@@@@@%*=-----===+=======------===-=  +@
# @@..@-........................ .+@@@@@@@==---=--==--=====================   @
# @#+@*........................ @@@@@@@@%%@@@@@@@@@@@%*====================.  @
# @@@@.......................-@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+===============-  @
# @@@.......................@%=.----##*=: .....:==+++#%@@@@@@@#==========+==  @
# @@:...........................---.........========++***#@@@@@@@*+======+++  @
# @%:..:..:::............................-=====#%@@@@@@@@%%#%@@@@@@#======*+..@
# @:::::::::::::.:::::..::.............====@@@@@@@@@@@@@@@@@@@@@@@@@@+=+++**..@
# #::::::::::::::::::::::::.........:===%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#+++**..@
# :::::#:::::::::::::::::::::::...-==+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#+++*..@
# :::+@:::::::::::::::::::::::::-==+@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@@@@@*++*..@
# ::@@:::::::::::::::::::::::::===@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:@@@@@@@@++*..@
# @@@::::::::::::::::::::::::===%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-@@@@@@@@*++:-@
# @@::::::::::::::::::::::::=++@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*+@@@@@@@@*::%@
# @+::::=::::::::::::::::::=+%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:@@@@@@@@*.:@@
# @::::=::::::::::::::::::++@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@:=@@@@@@@*::@@
# @@-:-@:::::::::::::::::=+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@::@@@@@@@*:%@@
# @@--@::::::::::::::::::+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*@@#-:@@@@@@=:@@@
# @@-@@:::---:::::::::::=@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+@@@--%@@@@@:%@@@
# @@@@-:-------------:--@@@@@@@@@@@@#@@@@@@@@@@@@@@@@@@@@@@@@@**@@---@@@@@:@@@@
# @@@@-----:-----------@@@@@+=@@@@@:@@@@@@@@@@@@@@@@@@@@@@@@@@@+@@%--=@@@%@@@@@
# @@@@-----:@---------%@@@%::@@@@@:@@@@@@@@@@@@@@@@@@@@@@%@@@@@+*@@---%@@@@@@@@
# @@@@----*@+--------=@@@:::@@@%@:-@@@@@@#@@@@@@@@@@@@@@@%@@@@@++@@*===@@@@@@@@
# @@@@@--@@@----:----@@=---+@@@%::@@@@@@#@@@@@@@@@@@%@@@@%@@@@@%++@%====@@@@@@@
# @@@@@@@@@----+----@%-----@@@@---@@@@@@+@@@@@*@@@@@*@@@@%#@@@@@++@@%===*@@@@@@
# @@@@@@@@#---*----+:-----@@@@---@@@@@@=#@@@@@#@@@@@*@@@@@#@@@@@*+*@@+==+@@@@@@
# @@@@@@@@---%@-----------@@@----@@@@@%=@@@@@*%@@@@%*@@@@@#%@@@@@*+@@@*=@@@@@@@
# @@@@@@@@--%@=----------@@@----@@@@@@==@@@@@*@@@@@#*@@@@@**@@@@@**%@@@@@@@@@@@
# @@@@@@@@@@@@--------===%@-----@@@@@===@@@@**@@@@@**#@@@@**%@@@@@**@@@@@@@@@@@
# @@@@@@@@@@@%-------%=-@@-----@@@@@@==#@@@@**@@@@@***@@@@**#@@@@@%+@@@@@@@@@@@
# @@@@@@@@@@@=-----=@=-=@-=----@@@@@===@@@@%**@@@@@***@@@@##*%@@@@@#@@@@@@@@@@@
# @@@@@@@@@@@-----*@@=-#-=====*@@@@*===@@@@**#@@@@@***#@@@%#+==@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@=-=%@@%=========@@@@@+==#@@@%**#@@@@@****@@@@++++@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@%@@@-=========@@@@++++%@@@***#@@@@@****@@@@++++@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@=======:::@@@%+++*@@@%**##@@@@@*===+@@@@++%@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@%======%:::=@@++++#@@@***##@@@@@#====@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@+===+@@:::#@*:::*%@@%***##==-@@@=++=@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@#:+@@::::#@@@****%====@@@%++@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@::::@@%%***%@++==@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@%@@@@@@%::%@@@****@@@+=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@:%@@@@@@@@@@%****@@@@@@@@@@@=@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@****@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%****@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                      
# """, end='')
    

    #%% load or initialize the result dictionary 
    if result_dict_in:
        result_dict = result_dict_in.copy()
    else:
        result_dict = {} #Add here a comprehensive constructor for the result dictionary
        result_dict['Upsilon_star_plus']    = {}
        result_dict['Upsilon_star_minus']   = {}
        
      
        
    #%%  set up      
    KSTAR_RANGE                        = range(20, K_M) # Range of kstar values to consider
    p                                  = len(Y) / (len(Y) + len(X))
    nX = X.shape[0]
    #%%   load or compute the null 
    if not stats_null:
        
        from utils_EE_v17 import compute_the_null
        
        stats_null                     = compute_the_null(p, K_M)
        
    #%%   compute the critical thresholds
    result_dict['stats_null']                 = stats_null
    result_dict['K_M']                        = K_M
    result_dict['p_ext']                      = p_ext
    result_dict['Upsilon_star_plus'][p_ext]   = np.quantile( stats_null[p],  1-p_ext )
    result_dict['Upsilon_star_minus'][p_ext]  = np.quantile( stats_null[1-p],  1-p_ext )
    
    #%%   compute the putative anomalie using binary sequences and binomial stats
    if not result_dict.get('Y^+') or not result_dict.get('X^+'):
        print("-----------------------------------------------------------------")
        print("Compute the nearest neighbours")
        
        from utils_EE_v17 import compute_nearest_neighbors
        
        indices, Knn_model             = compute_nearest_neighbors(X, Y, K_M, n_jobs=n_jobs)
        result_dict['Knn_model']       = Knn_model
        print("-----------------------------------------------------------------")
        print("Flagging of putative anomalous points")
        print("-----------------------------------------------------------------")  
        display(Math(r"\text{Compute} \ \mathcal{Y}^{+}"))
        binary_seq                     = (indices > X.shape[0]).astype(int)
        binary_seq[:X.shape[0], 0]     = 1                                     # injection 1 by 1
        p_val_info                     = PValueCalculator(binary_seq, KSTAR_RANGE, p=p)
        
        # Putative anomalies result_dict['Upsilon_i_Y']>=result_dict['Upsilon_star_plus'][p_ext]
        result_dict['Upsilon_i_Y']     = p_val_info.min_pval_plus[X.shape[0]:]
        result_dict['Y^+']             = np.where(result_dict['Upsilon_i_Y']>result_dict['Upsilon_star_plus'][p_ext])[0]
        # injected set
        result_dict['Upsilon_i_Y_inj'] = p_val_info.min_pval_plus[:X.shape[0]]
        result_dict['Y_underscore^+']  = np.where(result_dict['Upsilon_i_Y_inj']>result_dict['Upsilon_star_plus'][p_ext])[0]
        
        result_dict['Y_IDE']           = {}
        result_dict['Y_Pruned']        = {}
        display("DONE!")

        
        
        display(Math(r"\text{Compute} \ \mathcal{X}^{+}"))
        binary_seq_rev                 = (~(indices > X.shape[0])).astype(int)
        binary_seq_rev[X.shape[0]:, 0] = 1                                     # injection 1 by 1
        p_val_info                     = PValueCalculator(binary_seq_rev, KSTAR_RANGE, p=1-p)
        
        # Putative anomalies result_dict['Upsilon_i_X']>=result_dict['Upsilon_star_minus'][p_ext]
        result_dict['Upsilon_i_X']     = p_val_info.min_pval_plus[:X.shape[0]]
        result_dict['X^+']             = np.where(result_dict['Upsilon_i_X']>result_dict['Upsilon_star_minus'][p_ext])[0]
        
        # injected set
        result_dict['Upsilon_i_X_inj']  = p_val_info.min_pval_plus[X.shape[0]:]
        result_dict['X_underscore^+']  = np.where(result_dict['Upsilon_i_X_inj']>result_dict['Upsilon_star_minus'][p_ext])[0]
        
        
        result_dict['X_IDE']           = {}
        result_dict['X_Pruned']        = {}
        display("DONE!")

        
    #%% Iterative Density Equalization:
        
    # Condition 1 
    #
    # If the p_ext is smaller then something we already computed then reuse it otherweise start from scratch
    Condition_Y_1 = bool(np.any(result_dict['Upsilon_i_Y'] >= result_dict['Upsilon_star_plus'][p_ext]))
    # Condition 2
    #
    # Do it only if there are points greater then the trashold
    Condition_Y_2 = bool(p_ext >= min(result_dict['Upsilon_star_plus'].keys()) )
    # Condition 3 
    #
    # Possible flag for completelly separated sets
    auxx = (indices > X.shape[0])
    auxx[:X.shape[0],:] = ~auxx[:X.shape[0],:]
    auxx = auxx.astype(int)
    # TEMP HARDCODED, AT LEAST 10% of opposite set in the KNN to proceed with IDE
    Condition_3 = (auxx==0).sum()/(auxx==1).sum() > 0.1 # hardcoded for now
    if not Condition_3:
        print("-----------------------------------------------------------------")
        display("!!! Significant global density difference !!!")
        display("!!! IDE will not be computed !!!")
        print("-----------------------------------------------------------------")
    result_dict['Condition_3'] = Condition_3
    #%%
    if Condition_Y_1 & Condition_Y_2 & Condition_3:
        print("-----------------------------------------------------------------")
        print("Pruning via iterative density equalization (IDE)")
        print("-----------------------------------------------------------------")  
        display(Math(r"\text{Compute} \ \hat{\mathcal{Y}}^+"))
        result_dict['Y_IDE']           = IDE(
        result_dict['Y_IDE'],
        Y, 
        X, 
        result_dict['Upsilon_i_Y'] , 
        result_dict['Upsilon_star_plus'][p_ext],
        K_M,
        p,
        n_jobs,
        Knn_model,
        nX
        )
        display("DONE!")      
    #%%
        
    # Condition 1 
    #
    # If the p_ext is smaller then something we already computed then reuse it otherweise start from scratch
    Condition_X_1 = bool(np.any(result_dict['Upsilon_i_X'] >= result_dict['Upsilon_star_minus'][p_ext]))
    # Condition 2
    #
    # Do it only if there are points greater then the trashold
    Condition_X_2 = bool(p_ext >= min(result_dict['Upsilon_star_minus'].keys()) )
    # Condition 3 
    #
    # Possible flag for completelly separated sets
    #TBA
    if Condition_X_1 & Condition_X_2 & Condition_3:

        display(Math(r"\text{Compute} \ \hat{\mathcal{X}}^+"))
        result_dict['X_IDE']           = IDE(
        result_dict['X_IDE'],
        X, 
        Y, 
        result_dict['Upsilon_i_X'] , 
        result_dict['Upsilon_star_minus'][p_ext],
        K_M,
        1-p,
        n_jobs,
        Knn_model,
        nX
        )
        display("DONE!")         
#%%     get The Pruned Sets
    if result_dict['Y_IDE']:
        result_dict['Y_Pruned']    = {p_ext:get_indicies(result_dict['Upsilon_star_plus'][p_ext],result_dict['Y_IDE']) }

    if result_dict['X_IDE']:
        result_dict['X_Pruned']    = {p_ext:get_indicies(result_dict['Upsilon_star_minus'][p_ext],result_dict['X_IDE']) }

    return result_dict, stats_null

#%%

#%%
# ----------------------------------------------------------------------
# 4) Repêchage
# ----------------------------------------------------------------------

def Repechage(X,Y,result_dict,clusters,p_ext=1e-5,quant=0.01):
    print("-----------------------------------------------------------------")
    print("Repêchage")
    print("-----------------------------------------------------------------")  
    clusters_plus,clusters_minus = clusters
    nX = X.shape[0]
    EE_book = {
    "Y_OVER_clusters": {
        i: {"Putative": [], "Pruned": [], "Repechaged": [], "Background": []}
        for i in range(len(clusters_plus))
    },
    "X_OVER_clusters": {
        i: {"Putative": [], "Pruned": [], "Repechaged": [], "Background": []}
        for i in range(len(clusters_minus))
    }
    }   
    
    # if the two sets are majourly disjoint the Pruned and the Repêchage sets are NOT computed
    if  not result_dict['Condition_3']:
        print("-----------------------------------------------------------------")
        display("!!! Significant global density difference !!!")
        display("!!! Repêchage will not be computed !!!")
        print("-----------------------------------------------------------------")    
        # in this case save only the putatives
        ii = 0
        for cluster in clusters_plus:
            cluster_Y = cluster[cluster>=nX]-nX
            ii+=1
            print("alpha =", ii)
            EE_book['Y_OVER_clusters'][ii-1]['Putative']     = cluster_Y
        ii = 0
        for cluster in clusters_minus:
            
            cluster_X = cluster[cluster<nX]            
            ii+=1
            print("alpha =", ii)
            EE_book['X_OVER_clusters'][ii-1]['Putative']     = cluster_X
        # in this case exit from the function
        return EE_book
    else:
        
        ii = 0
        if result_dict['Y_Pruned']:
            display(Math(r"\text{Compute} \ \mathcal{Y}_\alpha^{\mathrm{anom}}"))
        
            for cluster in clusters_plus:
                
                cluster_X = cluster[cluster<nX]
                cluster_Y = cluster[cluster>=nX]-nX
                
                ii+=1
                print("alpha =", ii)
                # Get intersection of overdensity and cluster
                intersection           = list(set(result_dict['Y_Pruned'][p_ext]).intersection(set(cluster_Y)))
                if intersection:
                    # get the local new trashold
                    Upsilon_alpha_plus = np.quantile(result_dict['Upsilon_i_Y'][intersection],quant)
                    EE_book['Y_OVER_clusters'][ii-1]['Upsilon_alpha_plus']     = Upsilon_alpha_plus
                    EE_book['Y_OVER_clusters'][ii-1]['Putative']     = cluster_Y
                    EE_book['Y_OVER_clusters'][ii-1]['Pruned']       = intersection
                    EE_book['Y_OVER_clusters'][ii-1]['Repechaged']   = [x for x in cluster_Y if result_dict['Upsilon_i_Y'][x] >= Upsilon_alpha_plus]
                    EE_book['Y_OVER_clusters'][ii-1]['Background']   = [x for x in cluster_X if result_dict['Upsilon_i_Y_inj'][x] >= Upsilon_alpha_plus]
            display("DONE!") 
        else:
            display("!!! No Y-Overdensities found !!!")
            
        ii = 0
        if result_dict['X_Pruned']:
            display(Math(r"\text{Compute} \ \mathcal{X}_\alpha^{\mathrm{anom}}"))
        
            for cluster in clusters_minus:
                
                cluster_X = cluster[cluster<nX]
                cluster_Y = cluster[cluster>=nX]-nX
                
                ii+=1
                print("alpha =", ii)
                # Get intersection of overdensity and cluster
                intersection           = list(set(result_dict['X_Pruned'][p_ext]).intersection(set(cluster_X)))
                if intersection:
                    # get the local new trashold
                    Upsilon_alpha_minus = np.quantile(result_dict['Upsilon_i_X'][intersection],quant)
                    EE_book['X_OVER_clusters'][ii-1]['Upsilon_alpha_minus']     = Upsilon_alpha_minus
                    EE_book['X_OVER_clusters'][ii-1]['Putative']     = cluster_X
                    EE_book['X_OVER_clusters'][ii-1]['Pruned']       = intersection
                    EE_book['X_OVER_clusters'][ii-1]['Repechaged']   = [x for x in cluster_X if result_dict['Upsilon_i_X'][x] >= Upsilon_alpha_minus]
                    EE_book['X_OVER_clusters'][ii-1]['Background']   = [x for x in cluster_Y if result_dict['Upsilon_i_X_inj'][x] >= Upsilon_alpha_minus]
            display("DONE!")    
        else:
            display("!!! No X-Overdensities found !!!")
        return EE_book

