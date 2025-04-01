#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:48:17 2025

@author: johan
"""

###############
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

# Make sure EagleEye is on your Python path
sys.path.append('../../eagleeye')
import EagleEye
from utils_EE import compute_the_null, partitioning_function

# from Data_class1 import Data
from IPython.display import display
def lon_formatter(tick):
    """
    Format tick values to climate-style longitude labels.
    
    - For 0 or 360, returns "0°".
    - For ticks less than 180, returns "X°E".
    - For 180, returns "180°".
    - For ticks above 180, returns "(360 - X)°W".
    """
    tick = int(round(tick))
    if tick == 0 or tick == 360:
        return "0°"
    elif tick < 180:
        return f"{tick}°E"
    elif tick == 180:
        return "180°"
    else:
        return f"{360 - tick}°W"


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

#%% Decembre Jannuary Febbruary

data_name = "Air2m_northern_DJF"
# Determine the season based on the file name
season = "Winter" if 'DJF' in data_name else "Summer"

# Get the current working directory, which is where the notebook is running from
current_script_path = os.path.dirname(__file__)

# setup
grid_discretization = [37, 144]
scale = 360 / grid_discretization[1]
lat_indices = range(13, 29 + 1)
filters = ["Lati_area", "Longi_Gaussian"]
n_microstates = 180
data_extension = ".npy"

#%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
start = 90
clusterss = np.zeros((37,144,4))
longiss   = []
iloc=0
start_day_offset = 0
quantile_index = 0

# needed for loading and filtering the climate datasets
from Data_class1 import Data


#%% EagleEye hyperparameters

p       = .5

K_M     = 100

p_ext   = 1e-5

n_jobs  = 10

NN_for_plotting = 10

# z_score_threshold = 1.65
#%%
stats_null                     = compute_the_null(p=p, K_M=K_M)

#%%

for center_longitude in [330, 340]:
# Define the window width in degrees for filtering
    window_width = 60
    longi, window_size = get_longitude_indices(center_longitude, window_width)

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

    air2m_1999_2022 = (
        climate_data.AIR2M_filtered[90 + 2130 * 2 :, :, :]
        [:, climate_data.lati, :][:, :, climate_data.longi]
        .reshape((2130, -1))
        .copy()
        * climate_data.scale_sqrt
    )


#%%
    # ========== Future Analysis (1999-2022) ==========
    X_future = air2m_1951_1974
    Y_future = air2m_1999_2022
    
    result_dict, stats_null = EagleEye.Soar(X_future, Y_future, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})

    #%% Cluter the Putative anomalies
    
    clusters = partitioning_function(X_future,Y_future,result_dict,p_ext=p_ext,Z=2.65 )
    
    #%% Repêchage
    
    EE_book = EagleEye.Repechage(X_future,Y_future,result_dict,clusters,p_ext=1e-5)


    imax=result_dict['Upsilon_i_Y'].argmax()
    
    neighborhood_idx_pp = result_dict['Knn_model'].kneighbors(Y_future[imax, :][np.newaxis,:])[1]
    nX = X_future.shape[0]
    
    list_a1 = neighborhood_idx_pp[neighborhood_idx_pp>=nX]-nX
    
    # list of narest neigh of the max exceeding the crit tr Upsilon_+^*
    list_tot = [x for x in list_a1[:NN_for_plotting] if x in EE_book['Y_OVER_clusters'][0]['Repechaged']]
    
    clusterss[ :,:, iloc] = climate_data.AIR2M_filtered[90 + 2130 * 2:, :, :][list_tot ].mean(axis=0)
    iloc=iloc+1


#%%
# June July August
data_name = "Air2m_northern_JJA"

# Determine the season based on the file name
season = "Winter" if 'DJF' in data_name else "Summer"

# Constants
grid_discretization = [37, 144]
scale = 360 / grid_discretization[1]
lati = range(13, 29 + 1)
filters = ["Lati_area", "Longi_Gaussian"]
n_microstates = 180
data_extension = ".npy"

start = 90
longiss   = []
# iloc=0
from Data_class1 import Data
for center_longitude in [340, 180]:
    # Precompute longitude indices if cc is used
    window_width = 60
    longi, window_size = get_longitude_indices(center_longitude, window_width)

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

    air2m_1999_2022 = (
        climate_data.AIR2M_filtered[90 + 2130 * 2 :, :, :]
        [:, climate_data.lati, :][:, :, climate_data.longi]
        .reshape((2130, -1))
        .copy()
        * climate_data.scale_sqrt
    )

#%%
    # ========== Future Analysis (1999-2022) ==========
    X_future = air2m_1951_1974
    Y_future = air2m_1999_2022
    result_dict, stats_null = EagleEye.Soar(X_future, Y_future, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})

    #%% Cluter the Putative anomalies
    
    clusters = partitioning_function(X_future,Y_future,result_dict,p_ext=p_ext,Z=2.65 )
    
    #%% Repêchage
    
    EE_book = EagleEye.Repechage(X_future,Y_future,result_dict,clusters,p_ext=1e-5)

    # # get an index of max 
    imax=result_dict['Upsilon_i_X'].argmax()
    
    neighborhood_idx_pp = result_dict['Knn_model'].kneighbors(X_future[imax, :][np.newaxis,:])[1]
    nX = X_future.shape[0]

    list_a1 = neighborhood_idx_pp[neighborhood_idx_pp<nX]
    # list of narest neigh of the max exceeding the crit tr Upsilon_-^*
    list_tot = [x for x in list_a1[:NN_for_plotting] if x in EE_book['X_OVER_clusters'][0]['Repechaged']]

    clusterss[ :,:, iloc] = climate_data.AIR2M_filtered[start_day_offset : 2130 + start_day_offset, :, :][list_tot ].mean(axis=0)
    iloc=iloc+1


clusterss = clusterss[:,:,[2,3,0,1]]

clusterss[clusterss>10]=10
clusterss[clusterss<-10]=-10
#%%

# import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plot_utils import (get_longitude_indices, visualize_microstate_mean_AIR2M,
                        plot_csv_data)

# ---- Example CSV Data Functions ----
def load_and_average(filename, base_path="dati_clima"):
    filepath = os.path.join(base_path, filename)
    data = np.loadtxt(filepath, delimiter=",")
    return np.mean(data, axis=1)

def generate_data_from_csv():
    """
    Load eight CSV files and return a list of 4 panels.
    (Assumes UNDER values are multiplied by -1.)
    """
    lon = np.arange(0, 361, 10)
    panels = []
    # Panel 1: Summer '75-'98 (JJA)
    pos1 = load_and_average("JJA_ref_51_74_test_75_98_OVER.csv")
    neg1 = load_and_average("JJA_ref_51_74_test_75_98_UNDER.csv") * (-1)
    panels.append({'lon': lon, 'pos': pos1, 'neg': neg1,
                   'title': "Summer '75-'98 vs Summer '51-'74"})
    # Panel 2: Winter '75-'98 (DJF)
    pos2 = load_and_average("DJF_ref_51_74_test_75_98_OVER.csv")
    neg2 = load_and_average("DJF_ref_51_74_test_75_98_UNDER.csv") * (-1)
    panels.append({'lon': lon, 'pos': pos2, 'neg': neg2,
                   'title': "Winter '75-'98 vs Winter '51-'74"})
    # Panel 3: Summer '99-'22 (JJA)
    pos3 = load_and_average("JJA_ref_51_74_test_99_22_OVER.csv")
    neg3 = load_and_average("JJA_ref_51_74_test_99_22_UNDER.csv") * (-1)
    panels.append({'lon': lon, 'pos': pos3, 'neg': neg3,
                   'title': "Summer '99-'22 vs Summer '51-'74"})
    # Panel 4: Winter '99-'22 (DJF)
    pos4 = load_and_average("DJF_ref_51_74_test_99_22_OVER.csv")
    neg4 = load_and_average("DJF_ref_51_74_test_99_22_UNDER.csv") * (-1)
    panels.append({'lon': lon, 'pos': pos4, 'neg': neg4,
                   'title': "Winter '99-'22 vs Winter '51-'74"})
    return panels

# # ---- Main Script ----
# if __name__ == '__main__':
#     # Load CSV data for 4 panels.
csv_panels = generate_data_from_csv()

# Create a figure with 3 rows and 4 columns via GridSpec.
# First two rows: CSV panels (each panel spans 2 columns).
# Third row: 4 individual microstate panels.
fig = plt.figure(figsize=(24, 20))
gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 2],  hspace=0.1, wspace=0.1)

# ---- CSV Panels (First two rows) ----
ax_csv_1 = fig.add_subplot(gs[0, 0:2])
ax_csv_2 = fig.add_subplot(gs[0, 2:4])
ax_csv_3 = fig.add_subplot(gs[1, 0:2])
ax_csv_4 = fig.add_subplot(gs[1, 2:4])
csv_axes = [ax_csv_1, ax_csv_2, ax_csv_3, ax_csv_4]

for ax, panel in zip(csv_axes, csv_panels):
    plot_csv_data(ax, panel)
ax_csv_2.set_yticks([])
ax_csv_4.set_yticks([])
ax_csv_1.set_xticks([])
ax_csv_2.set_xticks([])

# ---- Microstate Panels (Third row): 4 panels (one per column) ----
ax_air_1 = fig.add_subplot(gs[2, 0])
ax_air_2 = fig.add_subplot(gs[2, 1])
ax_air_3 = fig.add_subplot(gs[2, 2])
ax_air_4 = fig.add_subplot(gs[2, 3])
air_axes = [ax_air_1, ax_air_2, ax_air_3, ax_air_4]

tbp_dummy = [clusterss[:,:,ij] for ij in range(4)]

longi_dummy = []
for center in [340, 180, 330, 340]:
    longi, _ = get_longitude_indices(center, 60)
    longi_dummy.append(longi)

# Dummy overlay colors and dummy max NLP values.
colors_dummy = ['darkmagenta', 'darkmagenta', 'darkorange', 'darkorange']
max_values_dummy = [35, 35, 35, 35]

for ax, tbp, longi, col, max_val in zip(air_axes, tbp_dummy, longi_dummy, colors_dummy, max_values_dummy):
    visualize_microstate_mean_AIR2M(ax, tbp, longi, col, max_val)

# Optional: Add a common y-axis label on the left.
fig.text(0.085, 2/3, 'anomalous days', va='center', rotation='vertical', fontsize=15)

# Optional: Add a legend at the top for the CSV panels.
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(facecolor=(1.0, 0.549, 0.0, 0.3), edgecolor='darkorange', linewidth=1.5,
                            label='novel temperature patterns')
blue_patch = mpatches.Patch(facecolor=(0.545, 0.0, 0.545, 0.3), edgecolor='darkmagenta', linewidth=1.5,
                            label='missing temperature patterns')
fig.legend(handles=[red_patch, blue_patch], loc='upper center',
           bbox_to_anchor=(.5, 0.98), ncol=1, frameon=False, fontsize=14)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for ax, label in zip([ax_csv_1, ax_csv_2, ax_csv_3, ax_csv_4, ax_air_1, ax_air_2, ax_air_3, ax_air_4], labels):
    ax.text(0.02, 0.95, label, transform=ax.transAxes,
            fontsize=21, fontweight='bold', va='top', ha='left')
#plt.tight_layout(rect=[0, 0, 1, 0.96])
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=-10, vmax=10)
cmap = plt.get_cmap('seismic')  # You can change this to any colormap you prefer
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # This dummy array is needed for the colorbar

# Create a new axis below the current plots for the colorbar.
cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])

# Create the colorbar
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

# Add a label to the colorbar.
cbar.set_label('Air2m Temperature Anomalies (°C)', fontsize=21,
               labelpad=10, fontweight='bold')
cbar.ax.xaxis.set_label_position('top')

# Set the ticks on the colorbar to go from -10 to 10 in steps of 5.
cbar.set_ticks(np.arange(-10, 11, 5))

# Customize tick parameters (size and thickness).
cbar.ax.tick_params(labelsize=21, width=2)

# Adjust the layout so that the subplots don't overlap with the colorbar.
plt.subplots_adjust(bottom=0.15)
plt.show()
