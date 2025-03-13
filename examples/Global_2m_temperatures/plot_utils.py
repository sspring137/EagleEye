#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:16:05 2025

@author: johan
"""

# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def get_longitude_indices(center_deg, window_width_deg, discretization_step=2.5):
    """
    Calculate the range of longitude indices based on the central longitude and window width.
    
    Parameters:
      center_deg (float): Central longitude (degrees)
      window_width_deg (float): Window width (degrees)
      discretization_step (float): Degrees per index (default 2.5)
      
    Returns:
      longi (list): List of indices.
      window_size (int): Number of indices.
    """
    total_points = int(360 / discretization_step)
    center_idx = int(center_deg / discretization_step) % total_points
    half_window_idx = int(window_width_deg / (2 * discretization_step))
    start_idx = (center_idx - half_window_idx) % total_points
    end_idx = (center_idx + half_window_idx + 1) % total_points
    if start_idx < end_idx:
        longi = list(range(start_idx, end_idx))
    else:
        longi = list(range(start_idx, total_points)) + list(range(0, end_idx))
    window_size = len(longi)
    return longi, window_size

def degree_to_longitude(deg):
    """
    Convert a degree value to a string with a degree symbol.
    (You can enhance this function to add 'E'/'W' if needed.)
    """
    return f"{deg}°"

# def visualize_microstate_mean_AIR2M(ax, tbp, longi, color_local, max_NLPval_NearNeigh):
#     """
#     Plot atmospheric data using Basemap with overlays.
    
#     Parameters:
#       ax : Matplotlib axis on which to plot.
#       tbp : 2D NumPy array of data (e.g. 37x144).
#       longi : List of longitude indices defining a region of interest.
#       color_local : A color (e.g. 'orange') used for an overlay.
#       max_NLPval_NearNeigh : A numeric value (e.g. number of days) to display in the title.
#     """
#     # Create a Basemap (orthographic projection centered at lat=90, lon=0)
#     m = Basemap(projection='ortho', lat_0=90, lon_0=0, resolution='l', ax=ax)
#     m.shadedrelief()
#     m.drawlsmask(land_color='silver', ocean_color='white', lakes=True)
#     m.drawcoastlines()
#     m.drawcountries()
#     m.drawparallels(np.arange(0, 91, 15), labels=[1, 0, 0, 0], fontsize=8, linewidth=1.5)
#     m.drawmeridians(np.arange(0, 360, 30), labels=[0, 0, 0, 1], fontsize=8, linewidth=1.5)
    
#     scale = 2.5
#     # Here we define a full range of x-values (for 144 grid points) – adjust as needed.
#     longi1 = list(range(0, 144)) + list(range(1))
#     y = np.arange(0, 91, scale)
#     x = np.arange(0, 361, scale)
#     X, Y = np.meshgrid(x, y)
    
#     # Set contour levels based on the maximum absolute value in tbp.
#     if np.abs(tbp.max()) < 17:
#         p1 = -15
#         p97 = 15
#     else:
#         p1 = -45
#         p97 = 45
#     dp = (p97 - p1) / 240
#     levels1 = np.arange(p1, p97, dp)
    
#     # Map the grid
#     x1, y1 = m(X, Y)
#     ax.contourf(x1, y1, tbp[:, longi1], levels=levels1, alpha=1, cmap='seismic')
    
#     # Plot vertical boundaries at the region defined by "longi"
#     x2, y2 = m(X[13:30, [longi[0], longi[-1]]], Y[13:30, [longi[0], longi[-1]]])
#     ax.plot(x2[:, 0], y2[:, 0], c='limegreen', linewidth=3.5)
#     ax.plot(x2[:, 1], y2[:, 1], c='limegreen', linewidth=3.5)
#     if longi[-1] < longi[0]:
#         x2, y2 = m(X[[13, 29], :][:, range(longi[0] - 145, longi[-1] + 1)], 
#                     Y[[13, 29], :][:, range(longi[0] - 145, longi[-1] + 1)])
#         ax.plot(x2[0, :], y2[0, :], c='limegreen', linewidth=3.5)
#         ax.plot(x2[1, :], y2[1, :], c='limegreen', linewidth=3.5)
#     else:
#         x2, y2 = m(X[[13, 29], :][:, range(longi[0], longi[-1] + 1)], 
#                     Y[[13, 29], :][:, range(longi[0], longi[-1] + 1)])
#         ax.plot(x2[0, :], y2[0, :], c='limegreen', linewidth=3.5)
#         ax.plot(x2[1, :], y2[1, :], c='limegreen', linewidth=3.5)
    
#     # Plot an extra overlay line with the specified color.
#     x2, y2 = m(X[[0, 3], :][:, :144], Y[[0, 3], :][:, :144])
#     ax.plot(x2[1, :], y2[1, :], c=color_local, linewidth=5)
    
#     # Set the title using the mid-point of the provided longitude indices.
#     mid_deg = int(longi[int(len(longi)/2)] * scale)
#     ax.set_title('Air2m_' + degree_to_longitude(mid_deg) +
#                  ': average over ' + str(max_NLPval_NearNeigh) + ' days')
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


def visualize_microstate_mean_AIR2M(ax, tbp, longi, color_local, max_NLPval_NearNeigh):
    """
    Plot atmospheric data using Basemap with specific overlays indicating different regions.
    - ax: Matplotlib axis object where the plot will be drawn.
    - tbp: 2D array of data to plot.
    - longi: List of longitude indices for specific plotting regions.
    - color_local: Color used for highlighting specific features.
    """
    m = Basemap(projection='ortho', lat_0=90, lon_0=0, resolution='l', ax=ax)
    m.shadedrelief()
    m.drawlsmask(land_color='silver', ocean_color='white', lakes=True)
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(0, 90 + 1, 15.), labels=[1, 0, 0, 0], fontsize=0, linewidth=1.5)
    m.drawmeridians(np.arange(0., 360, 30.), labels=[0, 0, 0, 1], fontsize=0, linewidth=1.5)

    scale = 2.5
    longi1 = list(range(0, 144)) + list(range(1))
    y = np.arange(0, 91, scale)
    x = np.arange(0,361, scale)
    # longi = (list(range(0,144)) + list(range(1)))#(list(range(120,144)) + list(range(0,121)))
    X, Y = np.meshgrid(x, y)

    if np.abs(tbp.max())<17:
        p1 = -10
        p97 = 10
        dp = (p97 - p1)/240
    else:
        p1 = -45
        p97 = 45
        dp = (p97 - p1)/240

    # dp = int((p97 - p1) / 1000)
    levels1 = np.arange(p1, p97, dp)
    x1, y1 = m(X, Y)
    CS = ax.contourf(x1, y1, tbp[:, longi1], levels=levels1, alpha=1, cmap='seismic')

    x2, y2 = m(X[13:30, [longi[0], longi[-1]]], Y[13:30, [longi[0], longi[-1]]])
    ax.plot(x2[:, 0], y2[:, 0], c='limegreen', linewidth=3.5)
    ax.plot(x2[:, 1], y2[:, 1], c='limegreen', linewidth=3.5)
    if longi[-1] < longi[0]:
        x2, y2 = m(X[[13, 29], :][:, range(longi[0]-144-1 , longi[-1]+ 1, 1)], Y[[13, 29], :][:, range(longi[0]-144-1, longi[-1] + 1, 1)])
        ax.plot(x2[0, :], y2[0, :], c='limegreen', linewidth=3.5)
        ax.plot(x2[1, :], y2[1, :], c='limegreen', linewidth=3.5)
    else:
        x2, y2 = m(X[[13, 29], :][:, range(longi[0], longi[-1] + 1, 1)], Y[[13, 29], :][:, range(longi[0], longi[-1] + 1, 1)])
        ax.plot(x2[0, :], y2[0, :], c='limegreen', linewidth=3.5)
        ax.plot(x2[1, :], y2[1, :], c='limegreen', linewidth=3.5)

    x2, y2 = m(X[[0, 3], :][:, range(0, 144, 1)], Y[[0, 3], :][:, range(0, 144, 1)])
    ax.plot(x2[1, :], y2[1, :], c=color_local, linewidth=5)

    # ax.set_title('Air2m_'+degree_to_longitude(int(longi[int(len(longi)/2)]*2.5)) + ': average over ' + str(max_NLPval_NearNeigh) + ' days')

def plot_csv_data(ax, data_panel):
    """
    Plot CSV-based data on a given axis.
    
    Expects data_panel to be a dictionary with keys:
      'lon' : 1D array of longitudes,
      'pos' : 1D array for the positive (OVER) values,
      'neg' : 1D array for the negative (UNDER) values,
      'title' : Title text for the panel.
    
    The positive curve is plotted in red and filled; the negative curve in blue and filled.
    """
    # Define x-ticks and labels (assumes longitudes 0 to 360 in steps of 10)
    xticks = np.arange(0, 361, 30)
    yticks = np.arange(-400, 300, 100)
    # xtick_labels = [f"{tick}°" for tick in xticks]
    xtick_labels = [lon_formatter(tick) for tick in xticks]
    lon = data_panel['lon']
    pos = data_panel['pos']
    neg = data_panel['neg']
    
    ax.plot(lon, pos, color='darkorange', linewidth=2)
    ax.fill_between(lon, pos, 0, where=(pos >= 0), color='darkorange', alpha=0.3)
    
    ax.plot(lon, neg, color='darkmagenta', linewidth=2)
    ax.fill_between(lon, neg, 0, where=(neg <= 0), color='darkmagenta', alpha=0.3)
    
    ax.axhline(0, color='black', linewidth=1)
    ax.text(0.07, 0.95, data_panel['title'],
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.set_xlim(0, 360)
    ax.set_ylim(-400, 250)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=12)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=12)
    # Add a twin y-axis (mirroring the primary) with no tick labels.
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
