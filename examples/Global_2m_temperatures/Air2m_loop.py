#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:03:39 2025

Author: sspringe

This script processes different climatological datasets (DJF, JJA), applies
data transformations (detrending, deseasonalization), and uses the EagleEye
library to detect anomalies over specified longitudes and time windows.
"""

import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

# Make sure EagleEye is on your Python path
sys.path.append('../../eagleeye')
import EagleEye_v7

from Data_class1 import Data
from IPython.display import display


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


# List of datasets to process
datasets = ["Air2m_northern_DJF", "Air2m_northern_JJA"]

# Loop through each dataset
for data_name in datasets:

    # Determine the season based on the file name
    if 'DJF' in data_name:
        season = "Winter"
    elif 'JJA' in data_name:
        season = "Summer"
    else:
        season = "Unknown"

    data_extension = ".npy"
    
    # Get the current directory of the script and navigate to the DATA directory
    current_script_path = os.path.dirname(__file__)
    data_path = os.path.join(current_script_path, 'DATA', data_name)

    # Grid settings and parameters
    grid_discretization = [37, 144]
    scale = 360 / grid_discretization[1]
    lat_indices = range(13, 29 + 1)  # 13 through 29 inclusive
    filters = ["Lati_area", "Longi_Gaussian"]
    n_microstates = 180

    # Time-windows and longitudes to loop over
    start_times = range(0, 91, 10)         # e.g., 0, 10, 20, ... 90
    center_longitudes = range(0, 360, 10)  # e.g., 0, 10, 20, ... 350

    # Initialize dictionaries to store past and future results
    results_past = {
        cl: {st: None for st in start_times} for cl in center_longitudes
    }
    results_future = {
        cl: {st: None for st in start_times} for cl in center_longitudes
    }

    # Main loop over longitude centers and start times
    for longitude_idx, center_lon_deg in enumerate(center_longitudes):
        for start_time_idx, start_day_offset in enumerate(start_times):
            
            # Define the window width in degrees for filtering
            window_width = 60
            longi, window_size = get_longitude_indices(center_lon_deg, window_width)

            # Create a Data object to handle loading and preprocessing
            climate_data = Data(
                current_script_path,
                data_name,
                grid_discretization,
                data_extension,
                scale,
                lat_indices,
                longi,
                window_size,
                season,
                filters,
                n_microstates
            )

            # Load and preprocess data
            climate_data.load_data()
            climate_data.detrend_and_deseasonalize_data()
            climate_data.get_blocked_days()
            climate_data.scale_sqrt, grid_filter = climate_data.calculate_distance_weights()

            # Prepare rolling slices of data (past, two future periods)
            air2m_1951_1974 = (
                climate_data.AIR2M_filtered[start_day_offset : 2130 + start_day_offset, :, :]
                [:, climate_data.lati, :][:, :, climate_data.longi]
                .reshape((2130, -1))
                .copy()
                * climate_data.scale_sqrt
            )
            air2m_1975_1998 = (
                climate_data.AIR2M_filtered[90 + 2130 : 90 + 2130 * 2, :, :]
                [:, climate_data.lati, :][:, :, climate_data.longi]
                .reshape((2130, -1))
                .copy()
                * climate_data.scale_sqrt
            )
            air2m_1999_2022 = (
                climate_data.AIR2M_filtered[90 + 2130 * 2 :, :, :]
                [:, climate_data.lati, :][:, :, climate_data.longi]
                .reshape((2130, -1))
                .copy()
                * climate_data.scale_sqrt
            )

            # General EagleEye parameters
            validation_size = air2m_1951_1974.shape[0]
            K_M = 100
            CRITICAL_QUANTILES = [1 - 1e-4, 1 - 1e-5]
            NUM_CORES = 10

            # ========== Past Analysis (1975-1998) ==========
            reference_data_past = air2m_1951_1974
            test_data_past = air2m_1975_1998

            # 1) SOAR detection
            soar_result_past = EagleEye_v7.Soar(
                reference_data_past,
                test_data_past,
                result_dict_in={},
                K_M=K_M,
                critical_quantiles=CRITICAL_QUANTILES,
                num_cores=NUM_CORES,
                validation=validation_size,
                partition_size=100
            )

            # 2) Identify clusters
            quantile_index = 0
            z_score_threshold = 1.65
            clusters_past = EagleEye_v7.partitian_function(
                reference_data_past,
                test_data_past,
                soar_result_past,
                soar_result_past['Upsilon_star_plus'][quantile_index],
                soar_result_past['Upsilon_star_minus'][quantile_index],
                K_M=K_M,
                Z=z_score_threshold
            )
#%%
            # 3) Derive the IV/IE dictionaries
            iv_ie_dict_past = EagleEye_v7.IV_IE_get_dict(
                clusters_past,
                soar_result_past,
                CRITICAL_QUANTILES[quantile_index],
                test_data_past,
                reference_data_past
            )
            results_past[center_lon_deg][start_day_offset] = iv_ie_dict_past
#%%
            # ========== Future Analysis (1999-2022) ==========
            reference_data_future = air2m_1951_1974
            test_data_future = air2m_1999_2022

            # 1) SOAR detection
            soar_result_future = EagleEye_v7.Soar(
                reference_data_future,
                test_data_future,
                result_dict_in={},
                K_M=K_M,
                critical_quantiles=CRITICAL_QUANTILES,
                num_cores=NUM_CORES,
                validation=validation_size,
                partition_size=100
            )

            # 2) Identify clusters
            clusters_future = EagleEye_v7.partitian_function(
                reference_data_future,
                test_data_future,
                soar_result_future,
                soar_result_future['Upsilon_star_plus'][quantile_index],
                soar_result_future['Upsilon_star_minus'][quantile_index],
                K_M=K_M,
                Z=z_score_threshold
            )
#%%
            # 3) Derive the IV/IE dictionaries
            iv_ie_dict_future = EagleEye_v7.IV_IE_get_dict(
                clusters_future,
                soar_result_future,
                CRITICAL_QUANTILES[quantile_index],
                test_data_future,
                reference_data_future
            )
            results_future[center_lon_deg][start_day_offset] = iv_ie_dict_future
            print('Eccolo')
#%%
    # -------------------------------------------------------------------------
    #  Save the results to file (Pickle)
    # -------------------------------------------------------------------------
    # Note: Adjust file naming conventions as needed.
    results_past_filename = f"{data_name}_results_past.pkl"
    results_future_filename = f"{data_name}_results_future.pkl"

    with open(os.path.join(current_script_path, results_past_filename), "wb") as f_past:
        pickle.dump(results_past, f_past)

    with open(os.path.join(current_script_path, results_future_filename), "wb") as f_future:
        pickle.dump(results_future, f_future)

    print(f"Finished processing {data_name}. Results saved to:")
    print(f"  {results_past_filename}")
    print(f"  {results_future_filename}")
