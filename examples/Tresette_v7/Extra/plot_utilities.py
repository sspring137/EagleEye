#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:57:35 2025

@author: sspringe
"""



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

def get_shades(num_clusters, cmap_name='Reds'):
    """
    Returns a list of distinct colors sampled from a given colormap.
    Example: get_shades(4, 'Reds') -> 4 different red shades.
    """
    cmap = plt.cm.get_cmap(cmap_name)
    # We'll pick values between 0.2 and 0.8 to avoid being too light/dark
    if cmap_name=='Oranges':
        return [cmap(v) for v in np.linspace(0.3, .7, num_clusters)]
    else:
        return [cmap(v) for v in np.linspace(0.5, 1, num_clusters)]

def plot_3d_ie_extra(IV_IE_dict, test_data, reference_data):
    """Example of a single-key plot for IE_extra, each cluster either red or blue
       depending on your original logic.  (Optional if you still want the first plot.)"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    over_clusters = IV_IE_dict["OVER_clusters"]
    under_clusters = IV_IE_dict["UNDER_clusters"]
    
    # Let's assume IE_extra => test_data for OVER, reference_data for UNDER (just as an example):
    over_color = get_shades(len(over_clusters), 'Oranges')
    under_color = get_shades(len(under_clusters), 'Blues')
    
    # OVER
    for idx, (i, cluster_data) in enumerate(over_clusters.items()):
        row_indices = cluster_data["IE_extra"]  # a list of ints
        xs, ys, zs = [], [], []
        for ridx in row_indices:
            x, y, z = test_data[ridx]
            xs.append(x); ys.append(y); zs.append(z)
        ax.scatter(xs, ys, zs, c=[over_color[idx]], marker='.',s=15, label=f"OVER {i}" if idx==0 else "")
    
    # UNDER
    for idx, (i, cluster_data) in enumerate(under_clusters.items()):
        row_indices = cluster_data["IE_extra"]
        xs, ys, zs = [], [], []
        for ridx in row_indices:
            x, y, z = reference_data[ridx]
            xs.append(x); ys.append(y); zs.append(z)
        ax.scatter(xs, ys, zs, c=[under_color[idx]], marker='.',s=15, label=f"UNDER {i}" if idx==0 else "")
        
    ax.set_title("First Plot: IE_extra Example")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


#%%

def plot_3d_second(IV_IE_dict, test_data, reference_data):
    """
    One 3D scatter plot with:
      - OVER clusters: 'From_test' => test_data => red shades
      - UNDER clusters: 'From_ref' => reference_data => blue shades
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    over_clusters = IV_IE_dict["OVER_clusters"]
    under_clusters = IV_IE_dict["UNDER_clusters"]
    
    n_over = len(over_clusters)
    n_under = len(under_clusters)
    
    # Different red shades for each OVER cluster
    over_Reds = get_shades(n_over, 'Reds')
    # Different blue shades for each UNDER cluster
    under_Blues = get_shades(n_under, 'Blues')
    
    # Plot OVER using 'From_test'
    for idx, (cluster_id, cluster_data) in enumerate(over_clusters.items()):
        row_indices = cluster_data["From_test"]
        xs, ys, zs = [], [], []
        for ridx in row_indices:
            x, y, z = test_data[ridx]  # from test_data
            xs.append(x)
            ys.append(y)
            zs.append(z)
        
        ax.scatter(xs, ys, zs, c=[over_Reds[idx]], marker='.',s=15, label=f"OVER {cluster_id}" if idx == 0 else "")
    
    # Plot UNDER using 'From_ref'
    for idx, (cluster_id, cluster_data) in enumerate(under_clusters.items()):
        row_indices = cluster_data["From_ref"]
        xs, ys, zs = [], [], []
        for ridx in row_indices:
            x, y, z = reference_data[ridx]  # from reference_data
            xs.append(x)
            ys.append(y)
            zs.append(z)
        
        ax.scatter(xs, ys, zs, c=[under_Blues[idx]], marker='.',s=15, label=f"UNDER {cluster_id}" if idx == 0 else "")
    
    ax.set_title("Second Plot: OVER→From_test (Red), UNDER→From_ref (Blue)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()



#%%

def plot_3d_third(IV_IE_dict, test_data, reference_data):
    """
    One 3D scatter plot with:
      - OVER clusters: 'From_ref' => reference_data => blue shades
      - UNDER clusters: 'From_test' => test_data => red shades
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    over_clusters = IV_IE_dict["OVER_clusters"]
    under_clusters = IV_IE_dict["UNDER_clusters"]
    
    n_over = len(over_clusters)
    n_under = len(under_clusters)
    
    # Different blue shades for each OVER cluster
    over_Blues = get_shades(n_over, 'Blues')
    # Different red shades for each UNDER cluster
    under_Reds = get_shades(n_under, 'Reds')
    
    # Plot OVER using 'From_ref'
    for idx, (cluster_id, cluster_data) in enumerate(over_clusters.items()):
        row_indices = cluster_data["From_ref"]
        xs, ys, zs = [], [], []
        for ridx in row_indices:
            x, y, z = reference_data[ridx]  # from reference_data
            xs.append(x)
            ys.append(y)
            zs.append(z)
        
        ax.scatter(xs, ys, zs, c=[over_Blues[idx]], marker='.',s=15, label=f"OVER {cluster_id}" if idx == 0 else "")
    
    # Plot UNDER using 'From_test'
    for idx, (cluster_id, cluster_data) in enumerate(under_clusters.items()):
        row_indices = cluster_data["From_test"]
        xs, ys, zs = [], [], []
        for ridx in row_indices:
            x, y, z = test_data[ridx]  # from test_data
            xs.append(x)
            ys.append(y)
            zs.append(z)
        
        ax.scatter(xs, ys, zs, c=[under_Reds[idx]], marker='.',s=15, label=f"UNDER {cluster_id}" if idx == 0 else "")
    
    ax.set_title("Third Plot: OVER→From_ref (Blue), UNDER→From_test (Red)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


#%%

# utility_plots.py


def plot_injected_anomalies_in_uniform_background(
    reference_data, 
    anomaly_sizes_u, 
    data_with_anomaly, 
    anomaly_sizes_o,
    title="Injected anomalies in Uniform background"
):
    """
    Plots:
      1) 'reference_data' background (silver),
      2) last anomalies in reference_data (navy),
      3) last anomalies in data_with_anomaly (firebrick),
    all in the 3D space [-100,100]^3 by default.

    Parameters
    ----------
    reference_data : np.ndarray
        Full reference dataset, shape (N, 3).
    anomaly_sizes_u : list or array
        Tells how many points in 'reference_data' are anomalies 
        (used to slice from the end).
    data_with_anomaly : np.ndarray
        Full dataset with injected anomalies, shape (M, 3).
    anomaly_sizes_o : list or array
        Tells how many points in 'data_with_anomaly' are anomalies 
        (used to slice from the end).
    title : str
        Plot title.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1) Non-anomalous reference background in silver
    ax.scatter(
        reference_data[:-np.array(anomaly_sizes_u).sum(), 0],
        reference_data[:-np.array(anomaly_sizes_u).sum(), 1],
        reference_data[:-np.array(anomaly_sizes_u).sum(), 2],
        c='silver',
        alpha=0.5,
        s=0.3
    )

    # 2) Reference_data anomalies in navy
    ax.scatter(
        reference_data[-np.array(anomaly_sizes_u).sum():, 0],
        reference_data[-np.array(anomaly_sizes_u).sum():, 1],
        reference_data[-np.array(anomaly_sizes_u).sum():, 2],
        c='navy',
        alpha=1,
        s=5
    )

    # 3) Anomalies from data_with_anomaly in firebrick
    ax.scatter(
        data_with_anomaly[-np.array(anomaly_sizes_o).sum():, 0],
        data_with_anomaly[-np.array(anomaly_sizes_o).sum():, 1],
        data_with_anomaly[-np.array(anomaly_sizes_o).sum():, 2],
        c='firebrick',
        alpha=1,
        s=5
    )

    # Fixed axis limits
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    
    ax.set_title(title)
    plt.show()


def plot_data_points_marked_as_putative_anomalies(
    data_with_anomaly,
    reference_data,
    res_new,
    title="Data points marked as putative anomalies"
):
    """
    Plots the points that pass certain threshold conditions (Upsilon) 
    in 'data_with_anomaly' (autumn colormap) and 'reference_data' (winter_r colormap).
    Axes are set to [-100,100]^3 by default.

    Parameters
    ----------
    data_with_anomaly : np.ndarray
        The dataset with injected anomalies, shape (M, 3).
    reference_data : np.ndarray
        The reference dataset, shape (N, 3).
    res_new : dict
        A dictionary containing 'stats', 'stats_reverse', 
        'Upsilon_star_plus', 'Upsilon_star_minus', etc.
    title : str
        Plot title.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Indices for over-threshold in data_with_anomaly
    idx_plot = (res_new['stats']['Upsilon_i_plus'] >= res_new['Upsilon_star_plus'][0])
    sc = ax.scatter(
        data_with_anomaly[idx_plot, 0],
        data_with_anomaly[idx_plot, 1],
        data_with_anomaly[idx_plot, 2],
        cmap='autumn',
        c=res_new['stats']['Upsilon_i_plus'][idx_plot],
        alpha=1,
        s=5,
        vmin=0
    )
    colorbar = fig.colorbar(sc, ax=ax)
    colorbar.set_label('Upsilon_overd')

    # Indices for over-threshold in reference_data (reverse logic)
    idx_plot2 = (res_new['stats_reverse']['Upsilon_i_plus'] >= res_new['Upsilon_star_minus'][0])
    sc2 = ax.scatter(
        reference_data[idx_plot2, 0],
        reference_data[idx_plot2, 1],
        reference_data[idx_plot2, 2],
        cmap='winter_r',
        c=res_new['stats_reverse']['Upsilon_i_plus'][idx_plot2],
        alpha=1,
        s=5,
        vmin=0
    )
    colorbar2 = fig.colorbar(sc2, ax=ax)
    colorbar2.set_label('Upsilon_underd')

    # Fixed axis limits
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)

    ax.set_title(title)
    plt.show()


def plot_points_extracted_by_iterative_equalization(
    data_with_anomaly,
    reference_data,
    res_new,
    threshold=0.9999,
    title="Points extracted by Iterative Equalization"
):
        # TODO - Ad functionality to give upsilon star as
    """
    Plots the points extracted from 'data_with_anomaly' and 'reference_data' 
    via iterative equalization, using a specified threshold. 
    Axes are set to [-100,100]^3 by default.

    Parameters
    ----------
    data_with_anomaly : np.ndarray
        The dataset with injected anomalies, shape (M, 3).
    reference_data : np.ndarray
        The reference dataset, shape (N, 3).
    res_new : dict
        Dictionary containing 'overdensities' and 'underdensities' keyed by thresholds.
    threshold : float
        The key used in res_new['overdensities'] and res_new['underdensities'].
    title : str
        Plot title.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    over_idx = res_new['overdensities'][threshold]   # indices in data_with_anomaly
    under_idx = res_new['underdensities'][threshold] # indices in reference_data

    ax.scatter(
        data_with_anomaly[over_idx, 0],
        data_with_anomaly[over_idx, 1],
        data_with_anomaly[over_idx, 2],
        c='firebrick',
        alpha=1,
        s=5
    )

    ax.scatter(
        reference_data[under_idx, 0],
        reference_data[under_idx, 1],
        reference_data[under_idx, 2],
        c='navy',
        alpha=1,
        s=5
    )

    # Fixed axis limits
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    
    plt.title(title)
    plt.show()
