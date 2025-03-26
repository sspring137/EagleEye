#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:23:36 2024

@author: Sebastian Springer (sspringe137)
"""

import numpy as np
import sys
sys.path.append('../../eagleeye')
import EagleEye as EagleEye


# import torch
from sklearn.preprocessing import StandardScaler
import os
import sys
import pickle
#%% load the data
all_data = np.load('./Data/LHC_data1p1M_new_features.npy')
scaler = StandardScaler()
standardized_data = all_data[:,:-1]/np.abs(all_data[:,:-1]).max(axis=0)
####################################################################################
# Define the parameters
num_cores             = 20
data_size             = 500000

# Read in the first command line argument as anomaly_possibility
anomaly_size          = int(sys.argv[1])
num_neighbors         = int(sys.argv[2])

# Define Km and KM
kstar_range           = range(20, num_neighbors)

# Get the indicies of anomalies where the last column of all_data==1
anomaly_idx = np.where(all_data[:,-1]==1)[0]
####################################################################################
# First get reference and test (mixed) samples 
reference_samples = standardized_data[:data_size,:].copy()
if anomaly_size == 0:
    mixed_samples = standardized_data[ - data_size - 100000 + anomaly_size : -100000 + anomaly_size , : ].copy()   
    lables_mix = np.zeros(data_size)
else:
    mixed_samples = np.concatenate((standardized_data[data_size:data_size*2-anomaly_size,:], standardized_data[-anomaly_size:,:]), axis=0)
    lables_mix = np.concatenate((np.zeros(data_size-anomaly_size), np.ones(anomaly_size)), axis=0)
####################################################################################
# Begin calls to EagleEye
Upsilon_star_plus                 = 14
res,_                             = EagleEye.Soar(reference_samples, mixed_samples, result_dict_in = {}, K_M = num_neighbors,n_jobs=num_cores,p_ext=1e-5,stats_null={0.5:[Upsilon_star_plus]} )

# Save the labels and samples!
res['stats'] = {}
res['stats']['lables_mix']        = lables_mix 
res['stats']['mixed_samples']     = mixed_samples 
res['stats']['reference_samples'] = reference_samples 

# Save the res dictionary to a pickle file
with open(f'./Results/LHC_EagleEye_res_{data_size}_anomaly_size_{anomaly_size}_kstar_range_{kstar_range[0]}_{kstar_range[-1]}.pkl', 'wb') as f:
    pickle.dump(res, f)
print(f"Data saved successfully as LHC_EagleEye_res_{data_size}_anomaly_size_{anomaly_size}_kstar_range_{kstar_range[0]}_{kstar_range[-1]}.pkl ")

    




