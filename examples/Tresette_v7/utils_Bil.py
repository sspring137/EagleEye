#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:27:14 2025

@author: sspringe
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting


def generate_dumbbell(
    n_points,
    sphere_radius=1.0,
    cylinder_radius=0.5,
    half_distance=2.0,
    fraction_sphere=0.4
):
    """
    Generates n_points in 3D arranged in a dumbbell shape.

    Parameters
    ----------
    n_points : int
        Total number of points to generate.
    sphere_radius : float
        Radius of each sphere at the ends.
    cylinder_radius : float
        Radius of the cylindrical bar connecting the spheres.
    half_distance : float
        Half the distance between the centers of the two spheres.
        (i.e., if the spheres are centered at (-d, 0, 0) and (d, 0, 0),
        then half_distance = d)
    fraction_sphere : float
        Fraction of total points to put in each sphere. For example, 
        0.4 means 40% in the left sphere, 40% in the right sphere.
        The remaining 20% will go in the cylinder.

    Returns
    -------
    points : (n_points, 3) ndarray
        Numpy array of 3D points forming the dumbbell shape.
    """

    # Number of points that go into each sphere
    n_sphere = int(fraction_sphere * n_points)
    # So total in spheres is 2 * n_sphere
    # The remainder go into the cylinder
    n_cylinder = n_points - 2 * n_sphere

    # --- Generate points for the left sphere ---
    # Center of left sphere at (-half_distance, 0, 0)
    sphere_left = sample_sphere(n_sphere, center=(-half_distance, 0, 0), radius=sphere_radius)
    
    # --- Generate points for the right sphere ---
    # Center of right sphere at (half_distance, 0, 0)
    sphere_right = sample_sphere(n_sphere, center=(half_distance, 0, 0), radius=sphere_radius)

    # --- Generate points for the cylinder ---
    # We'll generate points uniformly in a cylinder that spans
    # from x=-half_distance to x=half_distance
    cylinder = sample_cylinder(n_cylinder, 
                               x_min=-half_distance, 
                               x_max=half_distance, 
                               radius=cylinder_radius)

    # Combine all points into a single array
    points = np.vstack([sphere_left, sphere_right, cylinder])
    return points


def sample_sphere(n, center=(0.0, 0.0, 0.0), radius=1.0):
    """
    Samples n points uniformly inside a sphere of given radius and center.
    """
    # Use the method of sampling in 3D:
    # 1) Generate random directions on the unit sphere
    # 2) Generate radial distances scaled so volume is uniform (r ~ (U)^(1/3))
    c = np.array(center)
    
    # Random directions
    # theta in [0, 2π), phi in [0, π]
    theta = 2 * np.pi * np.random.rand(n)
    phi = np.arccos(1 - 2 * np.random.rand(n))  # or phi = np.pi * np.random.rand(n)
    
    # Coordinates on the unit sphere (for radius=1)
    x_unit = np.sin(phi) * np.cos(theta)
    y_unit = np.sin(phi) * np.sin(theta)
    z_unit = np.cos(phi)
    
    # Random radii (for uniform distribution inside the sphere)
    # Volume element => r^3 is uniformly distributed in [0, R^3]
    r = radius * np.random.rand(n) ** (1/3)
    
    # Scale unit directions by radii
    x = c[0] + r * x_unit
    y = c[1] + r * y_unit
    z = c[2] + r * z_unit
    
    return np.column_stack((x, y, z))


def sample_cylinder(n, x_min=-2.0, x_max=2.0, radius=0.5):
    """
    Samples n points uniformly in a right circular cylinder aligned along the x-axis.
    The cylinder extends from x_min to x_max and has the given radius in y-z plane.
    """
    # Length of the cylinder along the x-direction
    length = x_max - x_min
    
    # 1) Sample x coordinates uniformly
    x = x_min + length * np.random.rand(n)
    
    # 2) Sample radius in the y-z plane uniformly
    # We want uniform distribution *inside* a circle => r^2 is uniform in [0, radius^2]
    r = radius * np.sqrt(np.random.rand(n))
    angles = 2 * np.pi * np.random.rand(n)
    y = r * np.cos(angles)
    z = r * np.sin(angles)
    
    return np.column_stack((x, y, z))

def plot_dumbbell(points, title="3D Dumbbell Scatter Plot", point_size=1):
    """
    Plots the given 3D points in a scatter plot.

    Parameters
    ----------
    points : (n_points, 3) ndarray
        Numpy array of 3D points to plot.
    title : str
        Title of the plot.
    point_size : int or float
        Size of the scatter points.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=x, cmap='viridis', s=point_size, alpha=0.6)

    # Add color bar (optional)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('X-axis Value')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)

    # Set aspect ratio to 'auto' for better visualization
    ax.set_box_aspect([np.ptp(coord) for coord in [x, y, z]])  # ptp: peak to peak (range)

    plt.show()

def plot_data(reference_data, test_data, c1, c2, n_anom ):
    """
    Plots the given 3D points in a scatter plot.

    Parameters
    ----------
    points : (n_points, 3) ndarray
        Numpy array of 3D points to plot.
    title : str
        Title of the plot.
    point_size : int or float
        Size of the scatter points.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = reference_data[:, 0]
    y = reference_data[:, 1]
    z = reference_data[:, 2]

    # Create scatter plot
    scatter = ax.scatter(x, y, z, c='silver', s=1, alpha=0.3)


    # Extract x, y, z coordinates
    x = test_data[-n_anom:, 0]
    y = test_data[-n_anom:, 1]
    z = test_data[-n_anom:, 2]

    # Create scatter plot
    scatter = ax.scatter(x, y, z, c='firebrick' , s=15, alpha=0.6)

    # Add color bar (optional)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('X-axis Value')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Dumbell anomaly')

    # Set aspect ratio to 'auto' for better visualization
    # ax.set_box_aspect([np.ptp(coord) for coord in [x, y, z]])  # ptp: peak to peak (range)

    plt.show()


#%%

def generate_ref_test(n_ref, n_test, n_anom):
    
    
    reference_data = np.random.uniform(low=-10, high=10, size=(n_ref, 3) )
    test_data = np.random.uniform(low=-10, high=10, size=(n_test-n_anom, 3))

    
    n_points = n_anom
    points = generate_dumbbell(
        n_points=n_points, 
        sphere_radius=3.0, 
        cylinder_radius=.7,
        half_distance=7.0,
        fraction_sphere=0.3
    )
    
    test_data = np.concatenate( (test_data, points))

    return reference_data, test_data


#%%




def get_shades(num_clusters, cmap_name='Reds'):
    """
    Returns a list of distinct colors sampled from a given colormap.
    Example: get_shades(4, 'Reds') -> 4 different red shades.
    """
    cmap = plt.cm.get_cmap(cmap_name)
    # We'll pick values between 0.2 and 0.8 to avoid being too light/dark
    if cmap_name=='Reds':
        return [cmap(v) for v in np.linspace(0.3, .7, num_clusters)]
    else:
        return [cmap(v) for v in np.linspace(0.3, 1, num_clusters)]

def plot_3d_ie_extra(IV_IE_dict, test_data, reference_data):
    """Example of a single-key plot for IE_extra, each cluster either red or blue
       depending on your original logic.  (Optional if you still want the first plot.)"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    over_clusters = IV_IE_dict["OVER_clusters"]
    under_clusters = IV_IE_dict["UNDER_clusters"]
    
    # Let's assume IE_extra => test_data for OVER, reference_data for UNDER (just as an example):
    over_color = get_shades(len(over_clusters), 'Reds')
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
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
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
    colorbar.set_label('Upsilon_i')

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
    colorbar2.set_label('Upsilon_j(rev)')

    # Fixed axis limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    ax.set_title(title)
    plt.show()


def plot_points_extracted_by_iterative_equalization(
    data_with_anomaly,
    reference_data,
    res_new,
    threshold=0.9999,
    title="Points extracted by Iterative Equalization"
):
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
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    plt.title(title)
    plt.show()
