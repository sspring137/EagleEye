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

Authors: Sebastian Springer (sspringe137) and Alessandro Laio

--------------------------------------------------------------------------------
## Non-Commercial Academic and Research License (NCARL) v1.0

### Terms and Conditions

1. **Grant of License**: Permission is hereby granted, free of charge, to any person or organization obtaining a copy of this software to use, copy, modify, and distribute the software for academic research, educational purposes, and personal non-commercial projects, subject to the following conditions:

2. **Non-Commercial Use**: Non-commercial use includes any use that is not intended for or directed towards commercial advantage or monetary compensation. Examples include academic research, teaching, and personal experimentation.

3. **Acknowledgment**: Any publications or products that use the software must include the following acknowledgment:
   - "This software uses EagleEye developed by Sebastian Springer and Alessandro Laio at the International School for Advanced Studies (SISSA), Via Bonomea, 265, 34136 Trieste TS (Italy)."

4. **Modification and Distribution**: Users are allowed to modify and distribute the software for non-commercial purposes, provided they include this license with any distribution and acknowledge the original authors.

5. **No Warranty**: The software is provided "as-is" without any warranty of any kind.

### Contact Information

For commercial licensing, please contact Sebastian Springer at sebastian.springer@sissa.it and Alessandro Laio at laio@sissa.it or in person at the International School for Advanced Studies (SISSA), Via Bonomea, 265, 34136 Trieste TS (Italy).
--------------------------------------------------------------------------------

## Commercial License Agreement (CLA) v1.0

### Terms and Conditions

1. **Grant of License**: Permission is hereby granted to any person or organization obtaining a copy of this software for commercial use, provided they comply with the terms and conditions outlined in this agreement and pay the applicable licensing fees.

2. **Commercial Use Definition**: Commercial use includes any use intended for or directed towards commercial advantage or monetary compensation. This includes, but is not limited to, use in a commercial product, offering services with the software, or using the software in a revenue-generating activity.

3. **Licensing Fees**: The licensee agrees to negotiate and pay a licensing fee with the International School for Advanced Studies (SISSA), Via Bonomea, 265, 34136 Trieste TS (Italy), for commercial use of the software. Contact Sebastian Springer and Alessandro Laio for details on pricing and payment terms.

4. **Modification and Distribution**: Users are allowed to modify and distribute the software under the terms of this commercial license, provided they include this license with any distribution and acknowledge the original authors.

5. **Warranty**: The software is provided with a limited warranty as outlined in the commercial licensing agreement. Details of the warranty can be provided upon request.

### Contact Information

For licensing fees, terms, and support, please contact Sebastian Springer at sebastian.springer@sissa.it and Alessandro Laio at laio@sissa.it.
--------------------------------------------------------------------------------
"""
import numpy as np
from scipy.stats import binom
import pickle
import multiprocessing as mp
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
        # Calculate PMF using binomial distribution with the given p.
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
        Compute the p-value for a given statistic sum of binary values.
        :param stat_sum: The sum of the binary sequence for which p-value is calculated.
        :param k: Sequence length used for this calculation.
        :return: Tuple of p-value and its error estimate.
        """
        # Compute p-values by summing probabilities for more extreme outcomes.
        sum_probabilities1 = sum(prob for outcome, prob in self.pmf_results[k].items() if outcome <= stat_sum)
        sum_probabilities2 = sum(prob for outcome, prob in self.pmf_results[k].items() if outcome >= stat_sum)
        pval = min(sum_probabilities1, sum_probabilities2)
        error = np.sqrt(pval * (1 - pval) / k)  # Standard error for the p-value estimate
        return pval, error
    
    def compute_pval_array(self, kstar):
        """
        Computes an array of p-values for each unique statistic in the sequence of length kstar.
        :param kstar: Sequence length.
        :return: Numpy array of p-values.
        """
        # Calculate the sum of binary sequences for each subsequence.
        stat_sum_list = np.sum(self.binary_sequence[:, :kstar], axis=1)
        unique_results = np.unique(stat_sum_list)
        pval_dict = {stat_sum: self.compute_pval(stat_sum, kstar) for stat_sum in unique_results}
        pval_array = np.array([pval_dict[stat_sum] for stat_sum in stat_sum_list])
        return pval_array
    
    def compute_smallest_pval_info(self):
        """
        Identify the smallest p-value among all calculated p-values across different sequence lengths for each observation.
        :return: Dictionary containing the smallest p-values and their associated statistics.
        """
        smallest_pval_info = {'min_pval': [], 'pval_error': [], 'kstar_min_pval': [], 'stat_sum_list': []}
        for i in range(len(self.binary_sequence)):
            min_pval_indices = np.argmin([self.pval_array_dict[kstar][i][0] for kstar in self.kstar_range])  # Access only the p-value part
            kstar_min_pval = self.kstar_range[min_pval_indices]
            min_pval, error = self.pval_array_dict[kstar_min_pval][i]
            stat_sum_min_pval = np.sum(self.binary_sequence[i, :kstar_min_pval])
            
            smallest_pval_info['min_pval'].append(min_pval)
            smallest_pval_info['pval_error'].append(error)
            smallest_pval_info['kstar_min_pval'].append(kstar_min_pval)
            smallest_pval_info['stat_sum_list'].append(stat_sum_min_pval)
        
        return smallest_pval_info
    
    def save_results(self, filename):
        """
        Save the calculated smallest p-value information to a file using pickle.
        :param filename: String, the path to save the file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.smallest_pval_info, f)


class PValueCalculatorParallel(PValueCalculator):
    def __init__(self, binary_sequence, kstar_range, p=0.5, pmf_results=None, num_cores=10):
        """
        Initialize the PValueCalculatorParallel with a binary sequence, a range of k* values, optionally precomputed PMFs,
        and the number of cores for parallel processing.
        :param binary_sequence: A numpy array of binary values.
        :param kstar_range: A range or list of integers representing different sequence lengths for p-value calculations.
        :param p: The probability for the binomial distribution.
        :param pmf_results: Optional dictionary of precomputed PMFs for each k in kstar_range.
        :param num_cores: Integer, the number of cores to use for parallel processing.
        """
        self.binary_sequence = binary_sequence
        self.kstar_range = kstar_range
        self.p = p  # Probability for binomial distribution
        self.num_cores = num_cores
        
        self.pmf_results = pmf_results if pmf_results is not None else self.compute_pmf_results()
        
        kstar_chunks = self.chunk_kstar_range(kstar_range, self.num_cores)
        self.pval_array_dict = {}
        
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            future_to_kstar = {executor.submit(self._compute_pval_array_chunk, chunk): chunk for chunk in kstar_chunks}
            for future in as_completed(future_to_kstar):
                chunk = future_to_kstar[future]
                try:
                    result = future.result()
                    self.pval_array_dict.update(result)
                    print(f'Processing partition {kstar_chunks.index(chunk) + 1}/{len(kstar_chunks)} completed.')
                except Exception as exc:
                    print(f'Processing partition {kstar_chunks.index(chunk) + 1}/{len(kstar_chunks)} generated an exception: {exc}')
        
        self.smallest_pval_info = self.compute_smallest_pval_info()
    
    def _compute_pval_array_mp(self, kstar):
        """
        Helper function to compute p-value arrays for a given kstar in parallel.
        :param kstar: Sequence length.
        :return: Tuple of kstar and its corresponding p-value array.
        """
        return kstar, self.compute_pval_array(kstar)
    
    def _compute_pval_array_chunk(self, kstar_chunk):
        """
        Compute p-value arrays for a chunk of kstar values.
        :param kstar_chunk: A chunk (list) of kstar values.
        :return: Dictionary of p-value arrays for the chunk.
        """
        return {kstar: self.compute_pval_array(kstar) for kstar in kstar_chunk}
    
    def chunk_kstar_range(self, kstar_range, num_chunks):
        """
        Divide the kstar range into chunks for parallel processing.
        :param kstar_range: A range or list of kstar values.
        :param num_chunks: Integer, the number of chunks to divide the range into.
        :return: List of numpy arrays, each representing a chunk of kstar values.
        """
        chunk_size = len(kstar_range) // num_chunks
        kstar_chunks = [kstar_range[i:i + chunk_size] for i in range(0, len(kstar_range), chunk_size)]
        return kstar_chunks
