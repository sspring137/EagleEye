#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:31:09 2025

@author: johan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:04:26 2025

@author: johan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --------------------------
# Data Loading and Averaging Functions
# --------------------------
def load_and_average(filename, base_path="dati_clima"):
    """
    Load a CSV file (from base_path/filename) and return the average 
    over the columns (i.e. an array with one value per row).
    
    Assumes that the CSV file is comma‐delimited and contains only numeric data.
    """
    filepath = os.path.join(base_path, filename)
    data = np.loadtxt(filepath, delimiter=",")
    # Average over columns (axis=1). The resulting 1D array should match the number of rows.
    return np.mean(data, axis=1)

def generate_data_from_csv():
    """
    Load the eight CSV files from the subfolder 'dati_clima/' and create
    a list of four panel dictionaries, each containing:
      - 'lon': longitude values (assumed to be 0° to 360° in steps of 10°)
      - 'pos': the average OVER values (to be plotted in red)
      - 'neg': the average UNDER values (to be plotted in blue)
      - 'title': a panel title
    
    The grouping is as follows:
      Panel 1 (Summer '75-'98): uses the JJA files for test_75_98
         - pos: "JJA_ref_51_74_test_75_98_OVER.csv"
         - neg: "JJA_ref_51_74_test_75_98_UNDER.csv"
      
      Panel 2 (Winter '75-'98): uses the DJF files for test_75_98
         - pos: "DJF_ref_51_74_test_75_98_OVER.csv"
         - neg: "DJF_ref_51_74_test_75_98_UNDER.csv"
      
      Panel 3 (Summer '99-'22): uses the JJA files for test_99_22
         - pos: "JJA_ref_51_74_test_99_22_OVER.csv"
         - neg: "JJA_ref_51_74_test_99_22_UNDER.csv"
      
      Panel 4 (Winter '99-'22): uses the DJF files for test_99_22
         - pos: "DJF_ref_51_74_test_99_22_OVER.csv"
         - neg: "DJF_ref_51_74_test_99_22_UNDER.csv"
    """
    # Define the longitude array (assumed to be 0,10,...,360)
    lon = np.arange(0, 361, 10)
    
    panels = []
    # Panel 1: Summer '75-'98 (JJA)
    pos1 = load_and_average("JJA_ref_51_74_test_75_98_OVER.csv")
    neg1 = load_and_average("JJA_ref_51_74_test_75_98_UNDER.csv")*(-1)
    panels.append({
         'lon': lon,
         'pos': pos1,
         'neg': neg1,
         'title': "Summer '75-'98 vs Summer '51-'74"
    })
    
    # Panel 2: Winter '75-'98 (DJF)
    pos2 = load_and_average("DJF_ref_51_74_test_75_98_OVER.csv")
    neg2 = load_and_average("DJF_ref_51_74_test_75_98_UNDER.csv")*(-1)
    panels.append({
         'lon': lon,
         'pos': pos2,
         'neg': neg2,
         'title': "Winter '75-'98 vs Winter '51-'74"
    })
    
    # Panel 3: Summer '99-'22 (JJA)
    pos3 = load_and_average("JJA_ref_51_74_test_99_22_OVER.csv")
    neg3 = load_and_average("JJA_ref_51_74_test_99_22_UNDER.csv")*(-1)
    panels.append({
         'lon': lon,
         'pos': pos3,
         'neg': neg3,
         'title': "Summer '99-'22 vs Summer '51-'74"
    })
    
    # Panel 4: Winter '99-'22 (DJF)
    pos4 = load_and_average("DJF_ref_51_74_test_99_22_OVER.csv")
    neg4 = load_and_average("DJF_ref_51_74_test_99_22_UNDER.csv")*(-1)
    panels.append({
         'lon': lon,
         'pos': pos4,
         'neg': neg4,
         'title': "Winter '99-'22 vs Winter '51-'74"
    })
    
    return panels

# --------------------------
# Helper: Longitude Formatter
# --------------------------
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

# --------------------------
# Plotting Function
# --------------------------
def plot_data(data_panels):
    """
    Create a 2x2 grid of subplots with:
      - A legend placed at the top of the figure that mimics a title with legend‐like handles.
      - Top row panels: no x-axis tick labels.
      - Bottom row panels: x-axis ticks every 30° with climate-style labels and an x-axis label.
      - Each panel shows:
           * A red curve (pos; novel temperature patterns) and a blue curve (neg; missing temperature patterns),
             both filled.
           * An internal title.
      - A common y-axis label 'anomalous days' along the left.
      - A second y-axis is added to each subplot (identical to the primary) but without tick marks.
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 6), sharey=True)
    
    # Create legend handles that mimic the legend items:
    red_patch = mpatches.Patch(
        facecolor=(1, 0, 0, 0.3),  # red with alpha=0.3
        edgecolor='red',
        linewidth=1.5,
        label='novel temperature patterns'
    )
    blue_patch = mpatches.Patch(
        facecolor=(0, 0, 1, 0.3),  # blue with alpha=0.3
        edgecolor='blue',
        linewidth=1.5,
        label='missing temperature patterns'
    )
    
    # Place the legend at the top center of the figure.
    fig.legend(
        handles=[red_patch, blue_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.03),
        ncol=1,
        frameon=False,
        fontsize=14
    )
    
    # Define xticks (every 30°) and generate climate-style tick labels.
    xticks = np.arange(0, 361, 30)
    yticks = np.arange(-400, 200+1, 100)
    xtick_labels = [lon_formatter(tick) for tick in xticks]
    
    for idx, ax in enumerate(axs.flat):
        panel_data = data_panels[idx]
        lon = panel_data['lon']
        pos = panel_data['pos']
        neg = panel_data['neg']
        
        # Plot and fill the positive (red) curve.
        ax.plot(lon, pos, color='red', linewidth=2)
        ax.fill_between(lon, pos, 0, where=(pos >= 0), color='red', alpha=0.3)
        
        # Plot and fill the negative (blue) curve.
        ax.plot(lon, neg, color='blue', linewidth=2)
        ax.fill_between(lon, neg, 0, where=(neg <= 0), color='blue', alpha=0.3)
        
        # Draw a horizontal line at y=0.
        ax.axhline(0, color='black', linewidth=1)
        
        # Add the panel's internal title (positioned at the top left).
        ax.text(
            0.02, 0.95, panel_data['title'],
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
            ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        
        ax.set_ylim(-400, 250)
        ax.set_xlim(0, 360)
        
        # Set x-axis ticks.
        row = idx // 2  # 0 for top row, 1 for bottom row.
        ax.set_xticks(xticks)
        if row == 1:
            ax.set_xticklabels(xtick_labels, fontsize=12)
            ax.set_xlabel("Longitude", fontsize=15)
        else:
            ax.set_xticklabels([])  # Hide tick labels in the top row.
        ax.set_yticklabels(yticks,fontsize=15)
        # Add a second (twin) y-axis that mirrors the primary axis but hides its ticks.

    
    # Add a common y-axis label along the left side.
    fig.text(0.00, 0.5, 'anomalous days', va='center', rotation='vertical', fontsize=15)
    
    # Adjust layout to leave room for the top legend (title).
    plt.tight_layout(rect=[0, 0, 1, 0.95])


    plt.show()


data_panels = generate_data_from_csv()  # Load and average CSV data from dati_clima/
plot_data(data_panels)                  # Plot the data using the generated data
