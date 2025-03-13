#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:07:27 2025
Modified to show for each file (year/season) both the UNDER and the OVER curves
in one panel. These four merged panels (arranged as 2×2) occupy the top two rows,
and the third row is reserved for later panels.
@author: sspringe
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Increase default font sizes and line widths (optional)
plt.rcParams.update({
    'font.size': 14,          
    'axes.labelsize': 16,     
    'axes.titlesize': 18,     
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,    
    'legend.fontsize': 14,    
    'lines.linewidth': 2,     
    'lines.markersize': 8     
})

def degree_to_longitude(deg):
    """
    Converts a degree value (0-360) to a formatted longitude string.
    For example, 0 and 360 become "0°", values <= 180 get "°E", and values > 180
    are converted to a western longitude.
    """
    if deg == 0 or deg == 360:
        return "0°"
    elif deg <= 180:
        return f"{deg}°E"
    else:
        return f"{360-deg}°W"

def plot_counts(ax, results, label_over='Numb. OVER days', label_under='Numb. UNDER days'):
    """
    For each center longitude in the results dictionary, this function calculates:
      - the mean and std (over start_day_offset) of the IE_extra count from OVER_clusters.
      - the mean and std (over start_day_offset) of the IE_extra count from UNDER_clusters.
    
    A periodic extension is added (if the first center longitude is 0, a point at 360 is appended).
    
    Both curves (OVER in black and UNDER in blue) are then plotted on the given axis.
    """
    # Sorted center longitudes (as strings, then converted to float)
    center_lons = sorted(results.keys(), key=float)
    center_lons_arr = np.array(center_lons, dtype=float)
    
    over_means = []
    over_stds = []
    under_means = []
    under_stds = []
    
    # Loop over each center longitude
    for lon in center_lons:
        start_offsets = sorted(results[lon].keys())
        over_counts = []
        under_counts = []
        for sd in start_offsets:
            clusters_under = results[lon][sd]['UNDER_clusters']
            clusters_over  = results[lon][sd]['OVER_clusters']
            
            count_under = sum(
                len(cluster["IE_extra"])
                for cluster in clusters_under.values() 
                if cluster is not None and cluster.get("IE_extra") is not None
            )
            count_over = sum(
                len(cluster["IE_extra"])
                for cluster in clusters_over.values() 
                if cluster is not None and cluster.get("IE_extra") is not None
            )
            over_counts.append(count_over)
            under_counts.append(count_under)
        
        over_means.append(np.mean(over_counts))
        over_stds.append(np.std(over_counts))
        under_means.append(np.mean(under_counts))
        under_stds.append(np.std(under_counts))
    
    over_means = np.array(over_means)
    over_stds  = np.array(over_stds)
    under_means = np.array(under_means)
    under_stds  = np.array(under_stds)
    
    # Add periodic extension: if the first center is 0, append a point at 360.
    if center_lons_arr[0] == 0:
        center_lons_arr = np.append(center_lons_arr, 360)
        over_means = np.append(over_means, over_means[0])
        over_stds  = np.append(over_stds, over_stds[0])
        under_means = np.append(under_means, under_means[0])
        under_stds  = np.append(under_stds, under_stds[0])
    
    # Plot the OVER IE_extra counts (black)
    ax.plot(center_lons_arr, over_means, color='black', label=label_over)
    ax.fill_between(center_lons_arr, over_means - over_stds, over_means + over_stds, 
                    color='black', alpha=0.3)
    
    # Plot the UNDER IE_extra counts (blue)
    ax.plot(center_lons_arr, under_means, color='blue', label=label_under)
    ax.fill_between(center_lons_arr, under_means - under_stds, under_means + under_stds, 
                    color='blue', alpha=0.3)
    
    ax.set_xlabel('Center Longitude (deg)')
    ax.set_ylabel('Extracted days by IE')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ticks = np.arange(0, 361, 30)
    ax.set_xticks(ticks)
    ax.set_xticklabels([degree_to_longitude(deg) for deg in ticks])
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 410)
#------------------------------------------------------------------------------
# File information (four files as before)
#------------------------------------------------------------------------------
file_info = {
    'DJF: ref 51-74; test 75-98':   'Air2m_northern_DJF_results_past.pkl',
    'JJA: ref 51-74; test 75-98':   'Air2m_northern_JJA_results_past.pkl',
    'DJF: ref 51-74; test 99-22':   'Air2m_northern_DJF_results_future.pkl',
    'JJA: ref 51-74; test 99-22':   'Air2m_northern_JJA_results_future.pkl'
}

#------------------------------------------------------------------------------
# Create a figure with an outer GridSpec of 2 rows:
#   - The top block (outer[0]) will hold a merged 2×2 grid (4 panels) for the current 4 files.
#   - The bottom block (outer[1]) is reserved for 4 future panels (left empty for now).
#------------------------------------------------------------------------------
fig = plt.figure(figsize=(25, 16))
# Outer grid: 2 rows; let the top block be larger than the bottom block.
outer = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.3)

# Top block: subdivide into a 2×2 grid (each panel will show both OVER and UNDER curves)
gs_top = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0],
                                            wspace=0.3, hspace=0.3)
axes_top = [fig.add_subplot(gs_top[i, j]) for i in range(2) for j in range(2)]

# Bottom block: reserved for future use; here we create 4 axes and turn them off.
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                               wspace=0.3)
axes_bottom = [fig.add_subplot(gs_bottom[0, j]) for j in range(4)]
# for ax in axes_bottom:
    # ax.axis('off')  # leave these panels empty for now

#------------------------------------------------------------------------------
# Loop over the file_info items and plot the data in each top panel.
# Each panel shows both the OVER and UNDER curves.
#------------------------------------------------------------------------------
for ax, (panel_title, filename) in zip(axes_top, file_info.items()):
    print(f"Loading {filename} for {panel_title} ...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    plot_counts(ax, results)
    ax.set_title(panel_title)

plt.tight_layout()
plt.show()
