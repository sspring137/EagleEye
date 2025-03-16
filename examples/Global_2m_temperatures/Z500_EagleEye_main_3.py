#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:10:39 2024

@author: sspringe
"""

import numpy as np
import os
import sys
from Data_class1 import Data
from IPython.display import display
import pickle
import matplotlib.pyplot as plt

sys.path.append('../../eagleeye')
import EagleEye_v7


def get_longitude_indices(center_deg, window_width_deg, discretization_step=2.5):
    """
    Calculate the range of longitude indices based on the central longitude in degrees
    and the window width in degrees, considering the discretization step.
    
    Parameters:
    - center_deg (float): Central longitude in degrees.
    - window_width_deg (float): Total width of the window in degrees.
    - discretization_step (float): Degrees per index step, default is 2.5.
    
    Returns:
    - list: Indices representing the window of longitudes.
    - int: Size of the longitude window.
    """
    total_points = int(360 / discretization_step)
    
    # Convert degrees to index
    center_idx = int(center_deg / discretization_step) % total_points
    half_window_idx = int(window_width_deg / (2 * discretization_step))
    
    # Calculate the start and end indices
    start_idx = (center_idx - half_window_idx) % total_points
    end_idx = (center_idx + half_window_idx + 1) % total_points

    # Determine the range of longitudes
    if start_idx < end_idx:
        longi = list(range(start_idx, end_idx))
    else:
        # The range wraps around the circular buffer
        longi = list(range(start_idx, total_points)) + list(range(0, end_idx))

    # Calculate the size of the window for possible further use
    window_size = len(longi)
    
    return longi, window_size

# Datasets to process
datasets = ["Air2m_northern_DJF", "Air2m_northern_JJA"]

# Loop through each dataset
for data_name in datasets:
    # Determine the season based on the file name
    if 'DJF' in data_name:
        season = "Winter"
    elif 'JJA' in data_name:
        season = "Summer"
    
    data_extension = ".npy"
    
    # Construct the path to the file
    # Get the current directory of the script
    current_script_path = os.path.dirname(__file__)
    
    # Navigate to the DATA directory
    data_path = os.path.join(current_script_path, 'DATA', data_name)
    
    grid_discretization = [37, 144]
    scale = 360 / grid_discretization[1]
    lati = range(13, 29 + 1)
    filters = ["Lati_area", "Longi_Gaussian"]
    n_microstates = 180
    
    # # Function to compute the NLPval
    # def calculate_p_values(binary_sequence, kstar_range):
    #     p_val_info = EagleEye3.PValueCalculatorParallel(binary_sequence, kstar_range, num_cores=16).smallest_pval_info
    #     NLPval = -np.log(np.array(p_val_info['min_pval']))
    #     return NLPval, p_val_info['kstar_min_pval']
    
    # Prepare anomalies arrays
    # anomalies_numb_past = np.zeros((36, 2130, 10))
    # anomalies_numb_future = np.zeros((36, 2130, 10))
    
    # anomalies_numb_past_detr = np.zeros((36, 2130, 10))
    # anomalies_numb_future_detr = np.zeros((36, 2130, 10))
    
    Startt = range(0, 91, 10)
    center_longitude = range(0, 360, 10)
    
    results_past   = {cl: {st: None for st in Startt} for cl in center_longitude}
    results_future = {cl: {st: None for st in Startt} for cl in center_longitude}
    
    # Loop through each longitude center and start time
    for cc in range(36):
        for ss in range(10):
            start = Startt[ss]
            window_width = 60  # window width in degrees
            longi, window_size = get_longitude_indices(center_longitude[cc], window_width)
            
            Data_ = Data(current_script_path, data_name, grid_discretization, data_extension, scale, lati, longi, window_size, season, filters, n_microstates)
            Data_.load_data()
            Data_.detrend_and_deseasonalize_data()
            
            Data_.get_blocked_days()
            Data_.scale_sqrt, grid_filter = Data_.calculate_distance_weights()
            
            AIR2M_filtered_1951_74 = Data_.AIR2M_filtered[start:2130 + start, :, :][:, Data_.lati, :][:, :, Data_.longi].reshape((2130, -1)).copy() * Data_.scale_sqrt
            AIR2M_filtered_1975_98 = Data_.AIR2M_filtered[90 + 2130:90 + 2130 * 2, :, :][:, Data_.lati, :][:, :, Data_.longi].reshape((2130, -1)).copy() * Data_.scale_sqrt
            AIR2M_filtered_1999_22 = Data_.AIR2M_filtered[90 + 2130 * 2:, :, :][:, Data_.lati, :][:, :, Data_.longi].reshape((2130, -1)).copy() * Data_.scale_sqrt
            
            
            VALIDATION           = AIR2M_filtered_1951_74.shape[0]
            K_M                  = 100
            CRITICAL_QUANTILES   = [1-1E-4,1-1E-5]
            NUM_CORES            = 10
            
            #%% EE get NLPval for each point
            reference_data       = AIR2M_filtered_1951_74
            test_data            = AIR2M_filtered_1975_98
            
            res_detr_past        = EagleEye_v7.Soar(reference_data, test_data, result_dict_in={}, K_M=K_M, critical_quantiles=CRITICAL_QUANTILES,  num_cores=NUM_CORES, validation=VALIDATION, partition_size=100)
            #%% # Clustering the potential anomalies
            qt                           = 0
            Z                            = 1.65
            clusters                     = EagleEye_v7.partitian_function(reference_data,test_data,res_detr_past,res_detr_past['Upsilon_star_plus'][qt], res_detr_past['Upsilon_star_minus'][qt], K_M=K_M, Z=Z)
            
            clusters_plus,clusters_minus = clusters
            
            #%%  ´# get the background indixes
            IV_IE_dict_past = EagleEye_v7.IV_IE_get_dict(clusters,res_detr_past, CRITICAL_QUANTILES[qt],test_data,reference_data)

            results_past[cc][ss] = IV_IE_dict_past

#%%
#%%
#%%           
            #%% EE get NLPval for each point
            reference_data       = AIR2M_filtered_1951_74
            test_data            = AIR2M_filtered_1999_22
            
            res_detr_future      = EagleEye_v7.Soar(reference_data, test_data, result_dict_in={}, K_M=K_M, critical_quantiles=CRITICAL_QUANTILES,  num_cores=NUM_CORES, validation=VALIDATION, partition_size=100)
            #%% # Clustering the potential anomalies
            qt                           = 0
            Z                            = 1.65
            clusters                     = EagleEye_v7.partitian_function(reference_data,test_data,res_detr_future,res_detr_future['Upsilon_star_plus'][qt], res_detr_future['Upsilon_star_minus'][qt], K_M=K_M, Z=Z)
            
            clusters_plus,clusters_minus = clusters
            
            #%%  ´# get the background indixes
            IV_IE_dict_future = EagleEye_v7.IV_IE_get_dict(clusters,res_detr_future, CRITICAL_QUANTILES[qt],test_data,reference_data)
            
                        
            results_future[cc][ss] = IV_IE_dict_future   
            print('Eccolo')                          
                        
            
            
            
            # binary_sequences, neighbourhood_indexes = From_data_to_binary.create_binary_array_cdist(AIR2M_filtered_1975_98, AIR2M_filtered_1951_74, num_neighbors, num_cores)
            # NLPval, kstar_ = calculate_p_values(binary_sequences, kstar_range)
            # anomalies_numb_past_detr[cc, :, ss] = np.squeeze(np.array(NLPval))
            
            # binary_sequences, neighbourhood_indexes = From_data_to_binary.create_binary_array_cdist(AIR2M_filtered_1999_22, AIR2M_filtered_1951_74, num_neighbors, num_cores)
            # NLPval, kstar_ = calculate_p_values(binary_sequences, kstar_range)
            # anomalies_numb_future_detr[cc, :, ss] = np.squeeze(np.array(NLPval))
            
            # # Non-detrended
            # AIR2M_filtered_1951_74 = Data_.Z500[start:2130 + start, :, :][:, Data_.lati, :][:, :, Data_.longi].reshape((2130, -1)).copy() * Data_.scale_sqrt
            # AIR2M_filtered_1975_98 = Data_.Z500[90 + 2130:90 + 2130 * 2, :, :][:, Data_.lati, :][:, :, Data_.longi].reshape((2130, -1)).copy() * Data_.scale_sqrt
            # AIR2M_filtered_1999_22 = Data_.Z500[90 + 2130 * 2:, :, :][:, Data_.lati, :][:, :, Data_.longi].reshape((2130, -1)).copy() * Data_.scale_sqrt
            
            # binary_sequences, neighbourhood_indexes = From_data_to_binary.create_binary_array_cdist(AIR2M_filtered_1975_98, AIR2M_filtered_1951_74, num_neighbors, num_cores)
            # NLPval, kstar_ = calculate_p_values(binary_sequences, kstar_range)
            # anomalies_numb_past[cc, :, ss] = np.squeeze(np.array(NLPval))
            
            # binary_sequences, neighbourhood_indexes = From_data_to_binary.create_binary_array_cdist(AIR2M_filtered_1999_22, AIR2M_filtered_1951_74, num_neighbors, num_cores)
            # NLPval, kstar_ = calculate_p_values(binary_sequences, kstar_range)
            # anomalies_numb_future[cc, :, ss] = np.squeeze(np.array(NLPval))

    # # Save multiple arrays into a single .npz file with metadata
    # metadata = f"Air2m anomalies NLPval {season}"
    
    # # Save arrays and metadata
    # np.savez(f'anomalies_Air2m_data_NLPval_EE3_{season}.npz', 
    #           anomalies_numb_past_detr=anomalies_numb_past_detr,
    #           anomalies_numb_past=anomalies_numb_past,
    #           anomalies_numb_future_detr=anomalies_numb_future_detr,
    #           anomalies_numb_future=anomalies_numb_future,
    #           metadata=metadata)












