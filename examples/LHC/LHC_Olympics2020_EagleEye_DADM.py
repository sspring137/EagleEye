"""
@author: Sebastian Springer (sspringe137) and Andre Scaffidi (AndreScaffidi)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
import EagleEye as EagleEye
import From_data_to_binary

# import torch
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import os
import sys
#%% load the data
all_data = np.load('data/LHC_data1p1M_new_features.npy')
scaler = StandardScaler()
standardized_data = all_data[:,:-1]/np.abs(all_data[:,:-1]).max(axis=0)
####################################################################################
# Define the parameters
num_cores = 10
data_size = 500000
val_size  = int(1 * data_size) # Itteravely throw one point at a time fron the reference set to the test set for validation

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
do_validation = True    
if do_validation:
    validation                                 = list(range(val_size)) # Give indicies of reference samples to use for point wise validation inejctions. 
else:
    validation = None
binary_sequences                               = From_data_to_binary.create_binary_array_cdist(mixed_samples, reference_samples, num_neighbors, num_cores=10, validation=validation,partition_size=10)
display('binary done!')
stats                                          = EagleEye.calculate_p_values(binary_sequences, kstar_range, validation=validation,num_cores=10)
display('pvalues done!')

# Get the statisitcs: Upsilons and kstars for the test + validation
Upsilon_i, kstar_, Upsilon_i_Val, kstar_Val    = stats['Upsilon_i'],stats['kstar_'],stats['Upsilon_i_Val'],stats['kstar_Val']


# Save the variables to a single file
filename = f'results/LHC_EagleEye_results_{data_size}_anomaly_size_{anomaly_size}_kstar_range_{kstar_range[0]}_{kstar_range[-1]}_.npz'
np.savez(filename, 
            binary_sequences=binary_sequences, 
            mixed_samples=mixed_samples, 
            reference_samples=reference_samples, 
            kstar_=kstar_, 
            kstar_Val=kstar_Val,
            Upsilon_i=Upsilon_i, 
            Upsilon_i_Val=Upsilon_i_Val,
            lables_mix=lables_mix, 
            anomaly_size=anomaly_size)
display('save done!')

print(f"Data saved successfully as {filename}.")

    




