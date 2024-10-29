#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:12:53 2024

@author: sspringe
@author: AndreScaffidi
"""

import h5py
import pandas 
import numpy as np

######################################################################################################
# Specify the file path - relative to local directory

file_path              = 'events_anomalydetection_v2.features.h5'
df_features            = pandas.read_hdf(file_path)
# Feature engineering
df_features['|p|1']    = np.sqrt(np.sum(df_features[['pxj1', 'pyj1', 'pzj1']]**2, axis=1))
df_features['|p|2']    = np.sqrt(np.sum(df_features[['pxj2', 'pyj2', 'pzj2']]**2, axis=1))
df_features['tau21j1'] = df_features['tau2j1']/df_features['tau1j1']
df_features['tau21j2'] = df_features['tau2j2']/df_features['tau1j2']
df_features['tau32j1'] = df_features['tau3j1']/df_features['tau2j1']
df_features['tau32j2'] = df_features['tau3j2']/df_features['tau2j2']

new_features           = ['|p|1', '|p|2', 'tau21j1', 'tau21j2', 'tau32j1', 'tau32j2']
df_features            = df_features[new_features + [col for col in df_features.columns if col not in new_features]]

# Transform the DataFrame into a NumPy array
LHC_data1p1M           = df_features.to_numpy()[ :, [0,1,2,3,4,5,9,16,20] ]
# Print the shape of the NumPy array to verify
print(LHC_data1p1M.shape)
LHC_data1p1M[np.isnan(LHC_data1p1M)]                  = 0     # Andre suggestion

np.save('LHC_data1p1M_new_features.npy',LHC_data1p1M)