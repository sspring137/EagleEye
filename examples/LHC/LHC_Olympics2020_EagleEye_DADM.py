#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:23:36 2024

@author: Sebastian Springer (sspringe137)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from eagleeye import EagleEye
from eagleeye import From_data_to_binary

# import torch
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import os
import sys
#%% load the data
all_data = np.load('data/LHC_data1p1M_new_features.npy')

scaler = StandardScaler()

standardized_data = all_data[:,:-1]/np.abs(all_data[:,:-1]).max(axis=0)

#%%
num_cores = 10
data_size = 500000

anomaly_possibilities = [int(arg) for arg in sys.argv[1:]]
num_neighbors         = 1000
kstar_range           = range(4, num_neighbors)


def calculate_p_values(binary_sequence, kstar_range):
    p_val_info = EagleEye.PValueCalculatorParallel(binary_sequence, kstar_range,num_cores=num_cores).smallest_pval_info
    NLPval = -np.log(np.array(p_val_info['min_pval']))
    return NLPval, p_val_info['kstar_min_pval']

#%%
already_exist = False


if (not already_exist):
    for anomaly_size in anomaly_possibilities:
        
        reference_samples = standardized_data[:data_size,:].copy()
        if anomaly_size == 0:
            mixed_samples = standardized_data[ - data_size - 100000 + anomaly_size : -100000 + anomaly_size , : ].copy()   
            lables_mix = np.zeros(data_size)
        else:
            mixed_samples = np.concatenate((standardized_data[data_size:data_size*2-anomaly_size,:], standardized_data[-anomaly_size:,:]), axis=0)
            lables_mix = np.concatenate((np.zeros(data_size-anomaly_size), np.ones(anomaly_size)), axis=0)
            
        binary_sequences, neighbourhood_indexes = From_data_to_binary.create_binary_array_cdist(mixed_samples, reference_samples, num_neighbors, num_cores)
        display('binary done!')
        NLPval, kstar_ = calculate_p_values(binary_sequences, kstar_range)
        display('NLPval done!')

        idx_P = np.where((NLPval>17))[0]
        idx_TP = idx_P[ idx_P > data_size - anomaly_size ]

        idx_N = np.where((NLPval<=17))[0]
        idx_FN = idx_N[ idx_N > data_size - anomaly_size ]
        
        # Construct the filename based on the given parameters
        filename = f'results/LHC_EagleEye_results_{data_size}_anomaly_size_{anomaly_size}_kstar_range_{kstar_range[0]}_{kstar_range[-1]}_.npz'
        
        # Save the variables to a single file
        np.savez(filename, 
                 binary_sequences=binary_sequences, 
                 neighbourhood_indexes=neighbourhood_indexes, 
                 mixed_samples=mixed_samples, 
                 reference_samples=reference_samples, 
                 kstar_=kstar_, 
                 NLPval=NLPval, 
                 lables_mix=lables_mix, 
                 anomaly_size=anomaly_size)
        display('save done!')
        
        print(f"Data saved successfully as {filename}.")
        

elif already_exist:
    
        # Construct the filename based on the given parameters
        filename = f'results/LHC_EagleEye_results_{data_size}_anomaly_size_{anomaly_size}_kstar_range_{kstar_range[0]}_{kstar_range[-1]}_.npz'
    
        # Load the variables from the file
        loaded_data = np.load(filename)
        
        # Get the list of all keywords (variable names)
        keywords = loaded_data.files
        
        print("List of all keywords:", keywords)
        
        # Accessing the variables using the keywords
        for keyword in keywords:
            print(f"{keyword}:", loaded_data[keyword])
            





