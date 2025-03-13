#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:54:23 2024

@author: sspringe
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
from scipy.stats import binom
import pickle
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed

class PValueCalculator:
    def __init__(self, binary_sequence, kstar_range, p=0.5, pmf_results=None):
        """
        Initialize the PValueCalculator with a binary sequence, a range of k* values, and optionally precomputed PMFs.
        :param binary_sequence: A numpy array of binary values.
        :param kstar_range: A range or list of integers representing different sequence lengths for p-value calculations.
        :param p: The probability for the binomial distribution.
        :param pmf_results: Optional dictionary of precomputed PMFs for each k in kstar_range.
        """
        self.binary_sequence = binary_sequence
        self.kstar_range = kstar_range
        self.p = p  # Probability for binomial distribution
        
        self.pmf_results = pmf_results if pmf_results is not None else self.compute_pmf_results()
        self.pval_array_dict = {kstar: self.compute_pval_array(kstar) for kstar in kstar_range}
        self.smallest_pval_info = self.compute_smallest_pval_info()
    
    def compute_pmf(self, k):
        """
        Compute the PMF for a given sequence length k using the binomial distribution B(K, p).
        :param k: Integer, sequence length.
        :return: A dictionary with possible outcomes as keys and their probabilities as values.
        """
        pmf = dict(zip(range(k+1), binom.pmf(range(k+1), k, self.p)))
        return pmf
    
    def compute_pmf_results(self):
        """
        Precompute PMFs for each sequence length in kstar_range.
        :return: A dictionary with k as keys and corresponding PMFs as values.
        """
        return {k: self.compute_pmf(k) for k in self.kstar_range}
    
    def compute_pval(self, stat_sum, k):
        """
        Compute the minus-tail and plus-tail p-values for a given statistic sum of binary values.
        :param stat_sum: The sum of the binary sequence for which p-value is calculated.
        :param k: Sequence length used for this calculation.
        :return: pval_minus, error_minus, pval_plus, error_plus
        """
        # sum_probabilities1 is the cumulative probability of outcomes <= stat_sum (minus-tail).
        pval_minus = sum(prob for outcome, prob in self.pmf_results[k].items() if outcome <= stat_sum)
        # sum_probabilities2 is the cumulative probability of outcomes >= stat_sum (plus-tail).
        pval_plus = sum(prob for outcome, prob in self.pmf_results[k].items() if outcome >= stat_sum)
        
        error_minus = np.sqrt(pval_minus * (1 - pval_minus) / k)
        error_plus = np.sqrt(pval_plus * (1 - pval_plus) / k)

        return pval_minus, error_minus, pval_plus, error_plus
    
    def compute_pval_array(self, kstar):
        """
        Computes an array of p-values for each unique statistic in the sequence of length kstar.
        Returns a numpy array of shape (N, 4) where each row corresponds to:
        [pval_minus, error_minus, pval_plus, error_plus]
        """
        stat_sum_list = np.sum(self.binary_sequence[:, :kstar], axis=1)
        unique_results = np.unique(stat_sum_list)
        pval_dict = {stat_sum: self.compute_pval(stat_sum, kstar) for stat_sum in unique_results}
        pval_array = np.array([pval_dict[stat_sum] for stat_sum in stat_sum_list])
        return pval_array
    
    def compute_smallest_pval_info(self):
        """
        Identify the smallest minus-tail and plus-tail p-values among all calculated p-values 
        across different sequence lengths for each observation.
        
        Returns a dictionary:
        {
            'min_pval_minus': [], 'pval_error_minus': [], 'kstar_min_pval_minus': [],
            'min_pval_plus': [], 'pval_error_plus': [], 'kstar_min_pval_plus': [],
            'stat_sum_list_minus': [], 'stat_sum_list_plus': []
        }
        """
        n = len(self.binary_sequence)
        smallest_pval_info = {
            'min_pval_minus': [],
            'pval_error_minus': [],
            'kstar_min_pval_minus': [],
            'stat_sum_list_minus': [],
            'min_pval_plus': [],
            'pval_error_plus': [],
            'kstar_min_pval_plus': [],
            'stat_sum_list_plus': []
        }
        
        for i in range(n):
            # Extract minus and plus pvals for this observation across all kstar
            pvals_minus = [self.pval_array_dict[kstar][i][0] for kstar in self.kstar_range]  # pval_minus
            errors_minus = [self.pval_array_dict[kstar][i][1] for kstar in self.kstar_range] # error_minus
            pvals_plus = [self.pval_array_dict[kstar][i][2] for kstar in self.kstar_range]   # pval_plus
            errors_plus = [self.pval_array_dict[kstar][i][3] for kstar in self.kstar_range]  # error_plus
            
            # Find indices of min pvals
            min_idx_minus = np.argmin(pvals_minus)
            min_idx_plus = np.argmin(pvals_plus)
            
            kstar_min_pval_minus = self.kstar_range[min_idx_minus]
            kstar_min_pval_plus = self.kstar_range[min_idx_plus]
            
            min_pval_minus = pvals_minus[min_idx_minus]
            error_minus = errors_minus[min_idx_minus]
            min_pval_plus = pvals_plus[min_idx_plus]
            error_plus = errors_plus[min_idx_plus]

            # Compute stat sums at those kstar_min_pval
            stat_sum_minus = np.sum(self.binary_sequence[i, :kstar_min_pval_minus])
            stat_sum_plus = np.sum(self.binary_sequence[i, :kstar_min_pval_plus])
            
            smallest_pval_info['min_pval_minus'].append(min_pval_minus)
            smallest_pval_info['pval_error_minus'].append(error_minus)
            smallest_pval_info['kstar_min_pval_minus'].append(kstar_min_pval_minus)
            smallest_pval_info['stat_sum_list_minus'].append(stat_sum_minus)

            smallest_pval_info['min_pval_plus'].append(min_pval_plus)
            smallest_pval_info['pval_error_plus'].append(error_plus)
            smallest_pval_info['kstar_min_pval_plus'].append(kstar_min_pval_plus)
            smallest_pval_info['stat_sum_list_plus'].append(stat_sum_plus)
        
        return smallest_pval_info
    
    def save_results(self, filename):
        """
        Save the calculated smallest p-value information to a file using pickle.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.smallest_pval_info, f)


class PValueCalculatorParallel(PValueCalculator):
    def __init__(self, binary_sequence, kstar_range, p=0.5, pmf_results=None, num_cores=10):
        """
        Parallel version of the PValueCalculator.
        """
        self.binary_sequence = binary_sequence
        self.kstar_range = kstar_range
        self.p = p  # Probability for binomial distribution
        self.num_cores = num_cores
        
        self.pmf_results = pmf_results if pmf_results is not None else self.compute_pmf_results()
        
        kstar_chunks = self.chunk_kstar_range(kstar_range, self.num_cores)
        print(f'Processing {len(kstar_range)} kstar values in {len(kstar_chunks)} partitions.')
        print(kstar_chunks[0], kstar_chunks[-1])
        
        self.pval_array_dict = {}
        num_partitions = len(kstar_chunks)

        def process_and_store(future):
            chunk_idx, result = future.result()
            self.pval_array_dict.update(result)
            print(f'Processing partition for upsilon calculation {chunk_idx + 1}/{num_partitions} completed.')

        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            futures = [
                executor.submit(self._compute_pval_array_chunk_with_index, idx, chunk)
                for idx, chunk in enumerate(kstar_chunks)
            ]

            for future in as_completed(futures):
                process_and_store(future)

        self.smallest_pval_info = self.compute_smallest_pval_info()

    def _compute_pval_array_chunk_with_index(self, chunk_idx, kstar_chunk):
        result = {kstar: self.compute_pval_array(kstar) for kstar in kstar_chunk}
        return chunk_idx, result
    
    def chunk_kstar_range(self, kstar_range, num_chunks):
        chunk_size = len(kstar_range) // num_chunks
        if chunk_size == 0:
            return [kstar_range]
        kstar_chunks = [kstar_range[i:i + chunk_size] for i in range(0, len(kstar_range), chunk_size)]
        return kstar_chunks


def calculate_p_values(binary_sequence, kstar_range, num_cores=10, validation=None): 
    """
    Returns separate statistics for minus and plus tails. 
    If validation is provided, it splits arrays into training and validation sets.
    """
    p_val_info = PValueCalculatorParallel(binary_sequence, kstar_range, num_cores=num_cores, p=0.5).smallest_pval_info
    
    # Compute Upsilon for minus and plus tails
    Upsilon_i_minus = -np.log(np.array(p_val_info['min_pval_minus']))
    Upsilon_i_plus = -np.log(np.array(p_val_info['min_pval_plus']))
    
    # Prepare results dictionary
    statistics = {}
    statistics['Upsilon_i_minus'] = []
    statistics['kstar_minus'] = []
    statistics['Upsilon_i_plus'] = []
    statistics['kstar_plus'] = []
    statistics['Upsilon_i_Val_minus'] = []
    statistics['kstar_Val_minus'] = []
    statistics['Upsilon_i_Val_plus'] = []
    statistics['kstar_Val_plus'] = []

    if validation is not None:
        if isinstance(validation, int):
            len_val = validation
        else:
            len_val = len(validation)
        
        # Split arrays into training (everything before last len_val entries) and validation (last len_val entries)
        statistics['Upsilon_i_Val_minus'] = Upsilon_i_minus[-len_val:]
        statistics['kstar_Val_minus'] = p_val_info['kstar_min_pval_minus'][-len_val:]
        statistics['Upsilon_i_Val_plus'] = Upsilon_i_plus[-len_val:]
        statistics['kstar_Val_plus'] = p_val_info['kstar_min_pval_plus'][-len_val:]
        
        statistics['Upsilon_i_minus'] = Upsilon_i_minus[:-len_val]
        statistics['kstar_minus'] = p_val_info['kstar_min_pval_minus'][:-len_val]
        statistics['Upsilon_i_plus'] = Upsilon_i_plus[:-len_val]
        statistics['kstar_plus'] = p_val_info['kstar_min_pval_plus'][:-len_val]

    else:
        statistics['Upsilon_i_minus'] = Upsilon_i_minus
        statistics['kstar_minus'] = p_val_info['kstar_min_pval_minus']
        statistics['Upsilon_i_plus'] = Upsilon_i_plus
        statistics['kstar_plus'] = p_val_info['kstar_min_pval_plus']

    return statistics
