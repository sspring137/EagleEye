#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:59:54 2025

@author: sspringe
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_injected_anomalies_in_uniform_background(
    X, 
    anomaly_sizes_u, 
    data_with_anomaly, 
    anomaly_sizes_o,
    title="Anomalies in Uniform background"
):
    """
    Plots:
      1) 'X' background (silver),
      2) last anomalies in X (navy),
      3) last anomalies in data_with_anomaly (firebrick),
    all in the 3D space [-100,100]^3 by default.

    Parameters
    ----------
    X : np.ndarray
        Full reference dataset, shape (N, 3).
    anomaly_sizes_u : list or array
        Tells how many points in 'X' are anomalies 
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
        X[:-np.array(anomaly_sizes_u).sum(), 0],
        X[:-np.array(anomaly_sizes_u).sum(), 1],
        X[:-np.array(anomaly_sizes_u).sum(), 2],
        c='silver',
        alpha=0.5,
        s=0.3
    )

    # 2) X anomalies in navy
    ax.scatter(
        X[-np.array(anomaly_sizes_u).sum():, 0],
        X[-np.array(anomaly_sizes_u).sum():, 1],
        X[-np.array(anomaly_sizes_u).sum():, 2],
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

def plot_third_subplot(ax, 
                       Upsilon_i, 
                       null_distribution, 
                       Upsilon_star_plus, 
                       Upsilon_set_equalized, 
                       Upsilon_set_repechage,
                       Upsilon_set_pruned,
                       n_bins=100,
                       legend = True,
                       var_legend = 'X',
                       vlll = -50):
    """
    Plots a log-log histogram on the provided Axes object using pre-loaded data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which to plot.
    Upsilon_i : array-like
        Original dataset (e.g. first 50000 rows from column 0).
    null_distribution : array-like
        Dataset after equalization (e.g. all rows from column 1).
    Upsilon_star_plus : float
        Critical threshold value (e.g. first row, column 2).
    Upsilon_set_equalized : array-like
        Null distribution (e.g. first 47019 rows from column 3).
    Upsilon_set_repechage : array-like
        Points recognized as anomalies (e.g. first 2982 rows from column 4).
    n_bins : int, optional
        Number of logarithmically spaced bins (default is 100).
    """
    # -------------------------------
    # 1. Define fixed logarithmically spaced bins based on the Original dataset
    # -------------------------------
    bins = np.logspace(np.log10(np.min(Upsilon_i)),
                       np.log10(np.max(400)),
                       n_bins)

    # -------------------------------
    # 2. Compute histograms and normalize appropriately
    # -------------------------------
    # For the Original dataset, normalize by its own number of elements.
    counts_Upsilon_i, _ = np.histogram(Upsilon_i, bins=bins)
    norm_Upsilon_i = counts_Upsilon_i / len(Upsilon_i)
    
    # For the dataset after equalization, normalize by its own number of elements.
    counts_null_distribution, _ = np.histogram(null_distribution, bins=bins)
    norm_null_distribution = counts_null_distribution / len(null_distribution)
    
    # For the null distribution, normalize by its own number of elements.
    counts_Upsilon_set_equalized, _ = np.histogram(Upsilon_set_equalized, bins=bins)
    norm_Upsilon_set_equalized = counts_Upsilon_set_equalized / len(Upsilon_set_equalized)
    
    # For the anomalies, normalize by the size of the Original dataset.
    counts_Upsilon_set_repechage, _ = np.histogram(Upsilon_set_repechage, bins=bins)
    norm_Upsilon_set_repechage = counts_Upsilon_set_repechage / len(Upsilon_i)
    
    # For the anomalies, normalize by the size of the Original dataset.
    counts_Upsilon_set_pruned, _ = np.histogram(Upsilon_set_pruned, bins=bins)
    norm_Upsilon_set_pruned = counts_Upsilon_set_pruned / len(Upsilon_i)

    # Compute bin centers and widths (for bar plotting)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # Geometric mean for log bins.
    bin_widths  = np.diff(bins)                    # Width of each bin.

    # -------------------------------
    # 3. Plotting
    # -------------------------------
    # Set both axes to logarithmic scale.
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Increase tick thickness.
    ax.tick_params(axis='both', which='major', length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=1.5)

    # Plot Original dataset and anomalies as bars with white edges.
    #ax.bar(bin_centers, norm_Upsilon_i, width=bin_widths, color='silver',
    #       alpha=0.6, edgecolor='white', linewidth=2,
    #       label=r'Flagged anomalous points: $\mathcal{Y}^+$', align='center')
    # vlll = -50
        # First 32 bars with a white edge
    ax.bar(bin_centers[:vlll], norm_Upsilon_i[:vlll], width=bin_widths[:vlll],
           color='silver', alpha=0.6, edgecolor='white', linewidth=2,
            align='center')
    if var_legend =='X':
        # The remaining bars with a red edge
        ax.bar(bin_centers[vlll:], norm_Upsilon_i[vlll:], width=bin_widths[vlll:],
               color='silver', alpha=0.6, edgecolor='red', linewidth=2,
               label=r'Flagged anomalous points: $\mathcal{X}^+$',align='center')
        ax.bar(bin_centers, norm_Upsilon_set_repechage, width=bin_widths, color='limegreen',
               alpha=0.6, edgecolor='white', linewidth=2,
               label=r'Anomalies after $rep\hat{e}chage$: $\mathcal{X}_{\alpha}^{\mathrm{anom}}$', align='center')

        ax.bar(bin_centers, norm_Upsilon_set_pruned, width=bin_widths, color='darkgreen',
               alpha=0.6, edgecolor='white', linewidth=2,
               label=r'Pruned set: $\hat{\mathcal{X}}^+$', align='center')    



        # Plot the equalized dataset and null distribution as lines.
        ax.plot(bin_centers, norm_null_distribution, marker='.', color='black',
                label=r'Null distribution')

        ax.bar(bin_centers, norm_Upsilon_set_equalized, width=bin_widths, color='dodgerblue',
               alpha=0.6, edgecolor='white', linewidth=2,
               label=r'Dataset after equalization: $\mathcal{X}^{\mathrm{eq}}$', align='center')   

        # Add a vertical dashed red line at the critical threshold.
        ax.axvline(x=Upsilon_star_plus, color='red', linestyle='--', linewidth=2,
                   label=r'Critical threshold:$\mathcal{X}_+^{*}$')
    else:
    
    
            # The remaining bars with a red edge
        ax.bar(bin_centers[vlll:], norm_Upsilon_i[vlll:], width=bin_widths[vlll:],
               color='silver', alpha=0.6, edgecolor='red', linewidth=2,
               label=r'Flagged anomalous points: $\mathcal{Y}^+$',align='center')
        ax.bar(bin_centers, norm_Upsilon_set_repechage, width=bin_widths, color='limegreen',
               alpha=0.6, edgecolor='white', linewidth=2,
               label=r'Anomalies after $rep\hat{e}chage$: $\mathcal{Y}_{\alpha}^{\mathrm{anom}}$', align='center')

        ax.bar(bin_centers, norm_Upsilon_set_pruned, width=bin_widths, color='darkgreen',
               alpha=0.6, edgecolor='white', linewidth=2,
               label=r'Pruned set: $\hat{\mathcal{Y}}^+$', align='center')    



        # Plot the equalized dataset and null distribution as lines.
        ax.plot(bin_centers, norm_null_distribution, marker='.', color='black',
                label=r'Null distribution')

        ax.bar(bin_centers, norm_Upsilon_set_equalized, width=bin_widths, color='dodgerblue',
               alpha=0.6, edgecolor='white', linewidth=2,
               label=r'Dataset after equalization: $\mathcal{Y}^{\mathrm{eq}}$', align='center')   

        # Add a vertical dashed red line at the critical threshold.
        ax.axvline(x=Upsilon_star_plus, color='red', linestyle='--', linewidth=2,
                   label=r'Critical threshold:$\mathcal{Y}_+^{*}$')



    # Set the x and y limits.
    ax.set_xlim([1.5, 400])
    ax.set_ylim([1e-5, 1])

    # Labeling and legend.
    ax.set_xlabel(r'$\mathbf{\Upsilon}_i$')
    
    handles, labels = ax.get_legend_handles_labels()

    # Define the desired order of indices.
    # For example, if the plotting order (and default legend order) is:
    #  0: 'Original dataset'
    #  1: 'Points detected as anomalies'
    #  2: 'Points removed by density equalization'
    #  3: 'Null distribution'
    #  4: 'Dataset after equalization'
    #  5: 'Critical Threshold'
    #
    # and you want the legend to list them in the order:
    # 'Critical Threshold', 'Original dataset', 'Null distribution',
    # 'Dataset after equalization', 'Points removed by density equalization',
    # 'Points detected as anomalies', then your order list would be:
    desired_order = [0, 1, 2, 4, 3, 5 ]

    # Reorder the handles and labels according to the desired order.
    ordered_handles = [handles[i] for i in desired_order]
    ordered_labels = [labels[i] for i in desired_order]

    # Now create the legend with the new ordering.
    if legend:
        ax.legend(ordered_handles, ordered_labels)
    #ax.legend()
    
#%%

def plot_first_subplot(ax, 
                       axes1,
                       fig,
                       result_dict, 
                       Y,
                       X,
                       truncated_cmap_reds, 
                       truncated_cmap_purples,
                      ):
    
    
    # Plotting test data using truncated Reds
    if np.any(result_dict['Y^+']):
        sc1 = axes1.scatter(
            Y[result_dict['Y^+'], 0],
            Y[result_dict['Y^+'], 1],
            Y[result_dict['Y^+'], 2],
            c=result_dict['Upsilon_i_Y'][result_dict['Y^+']],
            cmap=truncated_cmap_reds,  # Use truncated Reds colormap
            label=r'Flagged anomalous points: $\mathcal{Y}^+$',
            alpha=1,
            s=5,
            vmin=result_dict['Upsilon_star_plus'][result_dict['p_ext']]
        )

    # Colorbar for the first scatter

        colorbar1 = fig.colorbar(sc1, ax=axes1, orientation='vertical', fraction=0.05, pad=0.03)
        colorbar1.set_label(r'$\mathbf{\Upsilon}_i(\mathcal{Y}^+)$')

    # Plotting reference data using truncated Purples

    sc2 = axes1.scatter(
        X[result_dict['X^+'], 0],
        X[result_dict['X^+'], 1],
        X[result_dict['X^+'], 2],
        c=result_dict['Upsilon_i_X'][result_dict['X^+']],
        label=r'Flagged anomalous points: $\mathcal{X}^+$',
        cmap=truncated_cmap_purples,  # Use truncated Purples colormap
        alpha=1,
        s=5,
        vmin=result_dict['Upsilon_star_minus'][result_dict['p_ext']]
    )
    colorbar2 = fig.colorbar(sc2, ax=axes1, orientation='horizontal', fraction=0.05, pad=0.03)
    colorbar2.set_label(r'$\mathbf{\Upsilon}_i(\mathcal{X}^+)$')


    #axes[0].set_title("$\\Upsilon_i^+ \geq \\Upsilon_*^+$ in Test and Reference ", fontsize=14)
    ax.legend(markerfirst=True, markerscale=3)
    
#%%

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True  # Use LaTeX for all text
import matplotlib.colors as mcolors


# --- Define a function to truncate a colormap ---
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Return a new colormap which is a subset of the given colormap.
    
    Parameters:
        cmap : matplotlib.colors.Colormap
            The original colormap.
        minval : float
            The lower bound of the new colormap (0 to 1).
        maxval : float
            The upper bound of the new colormap (0 to 1).
        n : int
            The number of discrete colors.
    
    Returns:
        new_cmap : matplotlib.colors.LinearSegmentedColormap
            The truncated colormap.
    """
    new_colors = cmap(np.linspace(minval, maxval, n))
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"truncated({cmap.name}, {minval:.2f}, {maxval:.2f})", new_colors)
    return new_cmap


def plot_37_article(
    EE_book,
    result_dict,
    X,
    Y,
    p,
    anomaly_sizes_u,
    anomaly_sizes_o,
    Upsilon_i_equalized_Y,
    Upsilon_i_equalized_X,
    save_name
):


    # --- Create truncated versions of 'Oranges' and 'BuPu' colormaps ---
    truncated_cmap_reds = truncate_colormap(plt.get_cmap('Oranges'), minval=0.3, maxval=1.0, n=100)
    truncated_cmap_purples = truncate_colormap(plt.get_cmap('BuPu'), minval=0.5, maxval=1.0, n=100)

    # --- Extract clusters from EE_book ---
    over_clusters = EE_book.get("Y_OVER_clusters", {})
    under_clusters = EE_book.get("X_OVER_clusters", {})

    # Compute indices for various density lists
    overdensities_IE_AG = [
        idx 
        for cluster_data in over_clusters.values() 
        for idx in cluster_data.get('Repechaged', [])
    ]
    overdensities_IE = [
        idx 
        for cluster_data in over_clusters.values() 
        for idx in cluster_data.get('Pruned', [])
    ]
    underdensities_IE_AG = [
        idx 
        for cluster_data in under_clusters.values() 
        for idx in cluster_data.get('Repechaged', [])
    ]
    underdensities_IE = [
        idx 
        for cluster_data in under_clusters.values() 
        for idx in cluster_data.get('Pruned', [])
    ]

    # Get the data arrays from result_dict
    data_all_o       = result_dict['Upsilon_i_Y']
    data_sub_o_IE_AG = data_all_o[overdensities_IE_AG]
    data_sub_o_IE    = data_all_o[overdensities_IE]

    data_all_u       = result_dict['Upsilon_i_X']
    data_sub_u_IE_AG = data_all_u[underdensities_IE_AG]
    data_sub_u_IE    = data_all_u[underdensities_IE]

    # Create the figure and subplots using GridSpec
    fig = plt.figure(figsize=(18, 26))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    
    # 3D subplots
    ax0a = fig.add_subplot(gs[0, 0], projection='3d')
    ax0  = fig.add_subplot(gs[0, 1], projection='3d')
    ax1  = fig.add_subplot(gs[1, 0], projection='3d')
    ax1b = fig.add_subplot(gs[1, 1], projection='3d')
    # 1D distribution subplots
    ax2  = fig.add_subplot(gs[2, 0])
    ax3  = fig.add_subplot(gs[2, 1])
    
    # Define subsets of data for the 3D scatter plots
    what_ref   = X[:-np.array(anomaly_sizes_u).sum(), :]
    what_over  = Y[-np.array(anomaly_sizes_o).sum():, :]
    what_under = X[-np.array(anomaly_sizes_u).sum():, :]

    # Plot background and contaminations on ax0a
    ax0a.scatter(what_ref[:,0], what_ref[:,1], what_ref[:,2], 
                 c='lightgray', s=5, alpha=0.2, label=r'Background')
    ax0a.scatter(what_over[:,0], what_over[:,1], what_over[:,2], 
                 c='darkorange', s=15, alpha=1, label=r'Contamination of $\mathcal{Y}$')
    ax0a.scatter(what_under[:,0], what_under[:,1], what_under[:,2], 
                 c='darkmagenta', s=15, alpha=1, label=r'Contamination of $\mathcal{X}$')
    
    # Add legend to one of the axes (using ax0)
    ax0.legend(markerscale=2)

    # Plot the first subplot using the helper function
    plot_first_subplot(ax0, 
    ax0,
    fig,
    result_dict, 
    Y,
    X,
    truncated_cmap_reds, 
    truncated_cmap_purples
    )

    # Process and plot "OVER" anomalies on ax1
    silver_over = result_dict['Y^+']
    silver_over = [x for x in silver_over if x not in overdensities_IE_AG]
    overdensities_IE_AG = [x for x in overdensities_IE_AG if x not in overdensities_IE]
    
    ax1.scatter(Y[silver_over, 0], Y[silver_over, 1], Y[silver_over, 2],
                edgecolor='dimgray', facecolor='dimgray', marker='.', s=5, alpha=0.4,
                label=r'Flagged anomalous points: $Y^+$')
    ax1.scatter(Y[overdensities_IE_AG, 0], Y[overdensities_IE_AG, 1], Y[overdensities_IE_AG, 2],
                c='limegreen', marker='*', s=7, alpha=0.6,
                label=r'Pruned set: $\hat{Y}^+$')
    ax1.scatter(Y[overdensities_IE, 0], Y[overdensities_IE, 1], Y[overdensities_IE, 2],
                c='darkgreen', marker='*', s=11, alpha=1,
                label=r'Anomalies after $rep\hat{e}chage$: $Y_{\alpha}^{\textrm{anom}}$')
    
    # Set limits for the 3D plot on ax1
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    ax1.set_zlim(-100, 100)

    # Process and plot "UNDER" anomalies on ax1b
    silver_under = result_dict['X^+']
    silver_under = [x for x in silver_under if x not in underdensities_IE_AG]
    underdensities_IE_AG = [x for x in underdensities_IE_AG if x not in underdensities_IE]
    
    ax1b.scatter(X[silver_under, 0], X[silver_under, 1], X[silver_under, 2],
                 edgecolor='dimgray', facecolor='dimgray', marker='.', s=5, alpha=0.4,
                 label=r'Flagged anomalous points: $X^+$')
    ax1b.scatter(X[underdensities_IE_AG, 0], X[underdensities_IE_AG, 1], X[underdensities_IE_AG, 2],
                 c='limegreen', marker='*', s=7, alpha=0.6,
                 label=r'Pruned set: $\hat{X}^+$')
    ax1b.scatter(X[underdensities_IE, 0], X[underdensities_IE, 1], X[underdensities_IE, 2],
                 c='darkgreen', marker='*', s=11, alpha=1,
                 label=r'Anomalies after $rep\hat{e}chage$: $\mathcal{X}_{\alpha}^{\textrm{anom}}$')
    
    # Set limits for the 3D plot on ax1b
    ax1b.set_xlim(-100, 100)
    ax1b.set_ylim(-100, 100)
    ax1b.set_zlim(-100, 100)
    
    # Reorder legend handles for clarity in ax1 and ax1b
    handles1, _ = ax1.get_legend_handles_labels()
    handles2, _ = ax1b.get_legend_handles_labels()
    
    new_handles1 = [handles1[0], handles1[2], handles1[1]]
    new_handles2 = [handles2[0], handles2[2], handles2[1]]
    
    new_labels1 = [
        r'Flagged anomalous points: $\mathcal{Y}^+$',
        r'Pruned set: $\hat{\mathcal{Y}}^+$',
        r'Anomalies after $rep\hat{e}chage$: $\mathcal{Y}_{\alpha}^{\mathrm{anom}}$'
    ]
    new_labels2 = [
        r'Flagged anomalous points: $\mathcal{X}^+$',
        r'Pruned set: $\hat{\mathcal{X}}^+$',
        r'Anomalies after $rep\hat{e}chage$: $\mathcal{X}_{\alpha}^{\mathrm{anom}}$'
    ]
    
    ax1.legend(new_handles1, new_labels1, markerscale=5)
    ax1b.legend(new_handles2, new_labels2, markerscale=5)
    
    # Plot the 1D distributions using the helper function for each subplot
    plot_third_subplot(
        ax2, 
        data_all_o, 
        result_dict['stats_null'][p], 
        result_dict['Upsilon_star_plus'][result_dict['p_ext']], 
        Upsilon_i_equalized_Y, 
        data_sub_o_IE_AG, 
        data_sub_o_IE,
        n_bins=300,
        legend=True,
        var_legend='Y'
    )
    
    plot_third_subplot(
        ax3, 
        data_all_u, 
        result_dict['stats_null'][1-p], 
        result_dict['Upsilon_star_minus'][result_dict['p_ext']], 
        Upsilon_i_equalized_X, 
        data_sub_u_IE_AG, 
        data_sub_u_IE,
        n_bins=300,
        legend=True,
        var_legend='X'
    )
    
    # Adjust layout and add annotations to each subplot
    plt.tight_layout()
    
    ax0a.annotate('A', xy=(0.02, 0.95), xycoords='axes fraction',
                  fontsize=21, fontweight='bold', va='top', ha='left')
    ax0.annotate('B', xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=21, fontweight='bold', va='top', ha='left')
    ax1.annotate('C', xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=21, fontweight='bold', va='top', ha='left')
    ax1b.annotate('D', xy=(0.02, 0.95), xycoords='axes fraction',
                  fontsize=21, fontweight='bold', va='top', ha='left')
    ax1.set_title(r'$\mathcal{X}$ reference, $\mathcal{Y}$ test ($\mathcal{Y}$-Overdensities)')
    ax1b.set_title(r'$\mathcal{Y}$ reference, $\mathcal{X}$ test ($\mathcal{X}$-Overdensities)')
    ax2.annotate('E', xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=21, fontweight='bold', va='top', ha='left')
    ax3.annotate('F', xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=21, fontweight='bold', va='top', ha='left')
    
    # Save the figure using the provided file name and display it
    plt.savefig(save_name, format='pdf')
    plt.show()

#%%


def plot_37_article_vanishing(
    EE_book,
    result_dict,
    X,
    Y,
    p,
    points_reference_ball,
    points_test_ball,
    Upsilon_i_equalized_Y,
    Upsilon_i_equalized_X,
    save_name
):


    # --- Create truncated versions of 'Oranges' and 'BuPu' colormaps ---
    truncated_cmap_reds = truncate_colormap(plt.get_cmap('Oranges'), minval=0.3, maxval=1.0, n=100)
    truncated_cmap_purples = truncate_colormap(plt.get_cmap('BuPu'), minval=0.5, maxval=1.0, n=100)

    # --- Extract clusters from EE_book ---
    over_clusters = EE_book.get("Y_OVER_clusters", {})
    under_clusters = EE_book.get("X_OVER_clusters", {})

    # Compute indices for various density lists
    overdensities_IE_AG = [
        idx 
        for cluster_data in over_clusters.values() 
        for idx in cluster_data.get('Repechaged', [])
    ]
    overdensities_IE = [
        idx 
        for cluster_data in over_clusters.values() 
        for idx in cluster_data.get('Pruned', [])
    ]
    underdensities_IE_AG = [
        idx 
        for cluster_data in under_clusters.values() 
        for idx in cluster_data.get('Repechaged', [])
    ]
    underdensities_IE = [
        idx 
        for cluster_data in under_clusters.values() 
        for idx in cluster_data.get('Pruned', [])
    ]

    # Get the data arrays from result_dict
    data_all_o       = result_dict['Upsilon_i_Y']
    data_sub_o_IE_AG = data_all_o[overdensities_IE_AG]
    data_sub_o_IE    = data_all_o[overdensities_IE]

    data_all_u       = result_dict['Upsilon_i_X']
    data_sub_u_IE_AG = data_all_u[underdensities_IE_AG]
    data_sub_u_IE    = data_all_u[underdensities_IE]

    fig = plt.figure(figsize=(21, 21))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # 3D subplots (ax0, ax1) ...
    ax0a = fig.add_subplot(gs[0, 0], projection='3d')
    ax0 = fig.add_subplot(gs[0, 1], projection='3d')
    ax1 = fig.add_subplot(gs[1, 0], projection='3d')
    #ax1b = fig.add_subplot(gs[1, 1], projection='3d')
    # 1D distribution subplot
    #ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # -------------------------------------------------------------------------
    # Continue with your plotting code:
    axes = [ax0a, ax0, ax1,ax3]
    
    # Define subsets of data for the 3D scatter plots
    what_ref   = X[:-1000, :]

    # Plot background and contaminations on ax0a
    ax0a.scatter(what_ref[:,0], what_ref[:,1], what_ref[:,2], 
                 c='lightgray', s=.1, alpha=0.2, label=r'Background')
    ax0a.scatter(points_reference_ball[:,0], points_reference_ball[:,1], points_reference_ball[:,2], 
                 c='darkorange', s=15, alpha=1, label="Reference Dataset (within Sphere)")
    ax0a.scatter(points_test_ball[:,0], points_test_ball[:,1], points_test_ball[:,2], 
                 c='darkmagenta', s=15, alpha=1, label="Test Dataset (within Sphere)")
    
    # Add legend to one of the axes (using ax0)
    ax0.legend(markerscale=2)

    # Plot the first subplot using the helper function
    plot_first_subplot(ax0, 
    ax0,
    fig,
    result_dict, 
    Y,
    X,
    truncated_cmap_reds, 
    truncated_cmap_purples
    )
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)
    ax0.set_zlim(-1, 1)
    # Process and plot "OVER" anomalies on ax1
    silver_over = result_dict['Y^+']
    silver_over = [x for x in silver_over if x not in overdensities_IE_AG]
    overdensities_IE_AG = [x for x in overdensities_IE_AG if x not in overdensities_IE]
    
    ax1.scatter(Y[silver_over, 0], Y[silver_over, 1], Y[silver_over, 2],
                edgecolor='dimgray', facecolor='dimgray', marker='.', s=5, alpha=0.4,
                label=r'Flagged anomalous points: $Y^+$')
    ax1.scatter(Y[overdensities_IE_AG, 0], Y[overdensities_IE_AG, 1], Y[overdensities_IE_AG, 2],
                c='limegreen', marker='*', s=7, alpha=0.6,
                label=r'Pruned set: $\hat{Y}^+$')
    ax1.scatter(Y[overdensities_IE, 0], Y[overdensities_IE, 1], Y[overdensities_IE, 2],
                c='darkgreen', marker='*', s=11, alpha=1,
                label=r'Anomalies after $rep\hat{e}chage$: $Y_{\alpha}^{\textrm{anom}}$')
    
    # Set limits for the 3D plot on ax1
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)

    # Process and plot "UNDER" anomalies on ax1b
    silver_under = result_dict['X^+']
    silver_under = [x for x in silver_under if x not in underdensities_IE_AG]
    underdensities_IE_AG = [x for x in underdensities_IE_AG if x not in underdensities_IE]
    
    axes[2].scatter(X[silver_under, 0], X[silver_under, 1], X[silver_under, 2],
                 edgecolor='dimgray', facecolor='dimgray', marker='.', s=5, alpha=0.4,
                 label=r'Flagged anomalous points: $X^+$')
    axes[2].scatter(X[underdensities_IE_AG, 0], X[underdensities_IE_AG, 1], X[underdensities_IE_AG, 2],
                 c='limegreen', marker='*', s=7, alpha=0.6,
                 label=r'Pruned set: $\hat{X}^+$')
    axes[2].scatter(X[underdensities_IE, 0], X[underdensities_IE, 1], X[underdensities_IE, 2],
                 c='darkgreen', marker='*', s=11, alpha=1,
                 label=r'Anomalies after $rep\hat{e}chage$: $\mathcal{X}_{\alpha}^{\textrm{anom}}$')
    
    # Set limits for the 3D plot on ax1b
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_zlim(-1, 1)
    
    # Reorder legend handles for clarity in ax1 and ax1b
    # handles1, _ = ax1.get_legend_handles_labels()
    handles2, _ = axes[2].get_legend_handles_labels()
    
    # new_handles1 = [handles1[0], handles1[2], handles1[1]]
    new_handles2 = [handles2[0], handles2[2], handles2[1]]
    
    # new_labels1 = [
    #     r'Flagged anomalous points: $\mathcal{Y}^+$',
    #     r'Pruned set: $\hat{\mathcal{Y}}^+$',
    #     r'Anomalies after $rep\hat{e}chage$: $\mathcal{Y}_{\alpha}^{\mathrm{anom}}$'
    # ]
    new_labels2 = [
        r'Flagged anomalous points: $\mathcal{X}^+$',
        r'Pruned set: $\hat{\mathcal{X}}^+$',
        r'Anomalies after $rep\hat{e}chage$: $\mathcal{X}_{\alpha}^{\mathrm{anom}}$'
    ]
    
    axes[2].legend(new_handles2, new_labels2, markerscale=5)
    
    # Plot the 1D distributions using the helper function for each subplot
    # plot_third_subplot(
    #     ax3, 
    #     data_all_o, 
    #     result_dict['stats_null'][p], 
    #     result_dict['Upsilon_star_plus'][result_dict['p_ext']], 
    #     Upsilon_i_equalized_Y, 
    #     data_sub_o_IE_AG, 
    #     data_sub_o_IE,
    #     n_bins=300,
    #     legend=True,
    #     var_legend='Y'
    # )
    
    plot_third_subplot(
        ax3, 
        data_all_u, 
        result_dict['stats_null'][1-p], 
        result_dict['Upsilon_star_minus'][result_dict['p_ext']], 
        Upsilon_i_equalized_X, 
        data_sub_u_IE_AG, 
        data_sub_u_IE,
        n_bins=100,
        legend=True,
        var_legend='X',
        vlll = -33
    )
    
    # Adjust layout and add annotations to each subplot
    plt.tight_layout()
    
    ax0a.annotate('A', xy=(0.02, 0.95), xycoords='axes fraction',
                  fontsize=21, fontweight='bold', va='top', ha='left')
    ax0.annotate('B', xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=21, fontweight='bold', va='top', ha='left')
    ax1.annotate('C', xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=21, fontweight='bold', va='top', ha='left')
    ax3.annotate('D', xy=(0.02, 0.95), xycoords='axes fraction',
                  fontsize=21, fontweight='bold', va='top', ha='left')
    ax1.set_title(r'$\mathcal{X}$ reference, $\mathcal{Y}$ test ($\mathcal{Y}$-Overdensities)')
    # axes[2].set_title(r'$\mathcal{Y}$ reference, $\mathcal{X}$ test ($\mathcal{X}$-Overdensities)')
    # ax2.annotate('E', xy=(0.02, 0.95), xycoords='axes fraction',
    #              fontsize=21, fontweight='bold', va='top', ha='left')
    # ax3.annotate('F', xy=(0.02, 0.95), xycoords='axes fraction',
    #              fontsize=21, fontweight='bold', va='top', ha='left')
    
    # Save the figure using the provided file name and display it
    plt.savefig(save_name, format='pdf')
    plt.show()


#%% table

def generate_anomaly_table(EE_book, clusters, 
                           anomaly_sizes_o=None, anomaly_sizes_u=None, 
                           total_array_size=50000):
    """
    Generate and display an HTML table summarizing anomaly statistics.
    
    Parameters
    ----------
    EE_book : dict
        Dictionary containing keys such as 'Y_OVER_clusters' and 'X_OVER_clusters'
        with anomaly data.
    clusters : tuple
        A tuple containing two arrays (or lists) of clusters:
        (clusters_plus, clusters_minus).
    anomaly_sizes_o : list, optional
        List of anomaly sizes for the overdensities (default is [50, 100, 200, 300, 500, 700, 900]).
    anomaly_sizes_u : list, optional
        List of anomaly sizes for the underdensities (default is [100, 300, 700]).
    total_array_size : int, optional
        Total size of the array from which anomalies are extracted (default is 50000).
        
    Returns
    -------
    combined_df : pandas.DataFrame
        A DataFrame containing the combined anomaly statistics.
        
    The function also displays the table as HTML with MathJax typesetting.
    """
    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML, Javascript

    # Default anomaly sizes if not provided
    if anomaly_sizes_o is None:
        anomaly_sizes_o = [50, 100, 200, 300, 500, 700, 900]
    if anomaly_sizes_u is None:
        anomaly_sizes_u = [100, 300, 700]

    # Unpack clusters
    clusters_plus, clusters_minus = clusters

    # -----------------------------
    # Compute anomaly indices for overdensities (o)
    # -----------------------------
    total_anomaly_points_o = sum(anomaly_sizes_o)
    start_index_o = total_array_size - total_anomaly_points_o

    anomaly_indices_o = []
    current_index = start_index_o
    for size in anomaly_sizes_o:
        indices = list(range(current_index, current_index + size))
        anomaly_indices_o.append(indices)
        current_index += size
    # Shift indices by total_array_size
    anomaly_indices_o_back = [[elem + total_array_size for elem in sublist] 
                           for sublist in anomaly_indices_o]

    # -----------------------------
    # Compute anomaly indices for underdensities (u)
    # -----------------------------
    total_anomaly_points_u = sum(anomaly_sizes_u)
    start_index_u = total_array_size - total_anomaly_points_u

    anomaly_indices_u = []
    current_index = start_index_u
    for size in anomaly_sizes_u:
        indices = list(range(current_index, current_index + size))
        anomaly_indices_u.append(indices)
        current_index += size
    # Copy for later use
    anomaly_indices_u_back = [[elem for elem in sublist] for sublist in anomaly_indices_u]

    # -----------------------------
    # Define anomaly columns and compute sorted order based on EE_book data
    # -----------------------------
    anomaly_columns = [
        r'$\mathcal{Y_{\alpha=0}}$', r'$\mathcal{Y_{\alpha=1}}$', r'$\mathcal{Y_{\alpha=2}}$', 
        r'$\mathcal{Y_{\alpha=3}}$', r'$\mathcal{Y_{\alpha=4}}$', r'$\mathcal{Y_{\alpha=5}}$', 
        r'$\mathcal{Y_{\alpha=6}}$', r'$\mathcal{X_{\alpha=0}}$', r'$\mathcal{X_{\alpha=1}}$', 
        r'$\mathcal{X_{\alpha=2}}$'
    ]

    # Compute an ordering from EE_book['Y_OVER_clusters']
    order = []
    for key in list(EE_book['Y_OVER_clusters'].keys()):
        pruned = np.array(EE_book['Y_OVER_clusters'][key]['Pruned'])
        # Compute the mean of pruned values that are at least 47000
        order.append(pruned[pruned >= 47000].mean())
    sorted_indices = sorted(range(len(order)), key=lambda i: order[i])

    # -----------------------------
    # Define statistic names (row labels) for the summary tables
    # -----------------------------
    stat_names = [
        "Added set:                 ",                      
        "Flagged: " + r"$\mathcal{Y}^+$ or $\mathcal{X}^+$",             
        "Pruned: " + r"$\hat{\mathcal{Y}}^+$ or $\hat{\mathcal{X}}^+$ ",  
        "Repechage: " + r"$\mathcal{Y}_{\alpha}^{anom}$ or $\mathcal{X}_{\alpha}^{anom}$",  
        "Injected: " + r"$\mathcal{Y}_{\alpha}^{inj}$ or $\mathcal{X}_{\alpha}^{inj}$"   
    ]

    # -----------------------------
    # Initialize the DataFrames with NaN values
    # -----------------------------
    df = pd.DataFrame(np.nan, index=stat_names, columns=anomaly_columns)
    df1 = pd.DataFrame(np.nan, index=stat_names, columns=anomaly_columns)

    # Combine anomaly sizes for all clusters (Over and Under)
    anomaly_sizes_all = anomaly_sizes_o + anomaly_sizes_u

    # -----------------------------
    # Populate the DataFrames with statistics from EE_book and clusters
    # -----------------------------
    counter = 0
    for anomaly in anomaly_columns:
        # Row 0: Injected anomaly size
        df.at[stat_names[0], anomaly] = anomaly_sizes_all[counter]
        df1.at[stat_names[0], anomaly] = anomaly_sizes_all[counter]
        
        if counter < 7:
            # For Overdensities (first 7 columns)
            df.at[stat_names[1], anomaly] = len(clusters_plus[sorted_indices[counter]][
                clusters_plus[sorted_indices[counter]] >= 50000])
            df.at[stat_names[2], anomaly] = len(EE_book['Y_OVER_clusters'][sorted_indices[counter]]['Pruned'])
            df.at[stat_names[3], anomaly] = len(EE_book['Y_OVER_clusters'][sorted_indices[counter]]['Repechaged'])
            df.at[stat_names[4], anomaly] = len(EE_book['Y_OVER_clusters'][sorted_indices[counter]]['Background'])
            
            df1.at[stat_names[1], anomaly] = len([x for x in clusters_plus[sorted_indices[counter]][
                clusters_plus[sorted_indices[counter]] >= 50000] if x in anomaly_indices_o_back[counter]])
            df1.at[stat_names[2], anomaly] = len([x for x in EE_book['Y_OVER_clusters'][sorted_indices[counter]]['Pruned']
                                                  if x in anomaly_indices_o[counter]])
            df1.at[stat_names[3], anomaly] = len([x for x in EE_book['Y_OVER_clusters'][sorted_indices[counter]]['Repechaged']
                                                  if x in anomaly_indices_o[counter]])
            df1.at[stat_names[4], anomaly] = 0
        else:
            # For Underdensities (last 3 columns; adjust index by subtracting 7)
            idx = counter - 7
            df.at[stat_names[1], anomaly] = len(clusters_minus[idx][clusters_minus[idx] < 50000])
            df.at[stat_names[2], anomaly] = len(EE_book['X_OVER_clusters'][idx]['Pruned'])
            df.at[stat_names[3], anomaly] = len(EE_book['X_OVER_clusters'][idx]['Repechaged'])
            df.at[stat_names[4], anomaly] = len(EE_book['X_OVER_clusters'][idx]['Background'])
            
            df1.at[stat_names[1], anomaly] = len([x for x in clusters_minus[idx][clusters_minus[idx] < 50000]
                                                   if x in anomaly_indices_u_back[idx]])
            df1.at[stat_names[2], anomaly] = len([x for x in EE_book['X_OVER_clusters'][idx]['Pruned']
                                                   if x in anomaly_indices_u[idx]])
            df1.at[stat_names[3], anomaly] = len([x for x in EE_book['X_OVER_clusters'][idx]['Repechaged']
                                                   if x in anomaly_indices_u[idx]])
            df1.at[stat_names[4], anomaly] = 0
        counter += 1

    # -----------------------------
    # Combine the two DataFrames into one summary table
    # -----------------------------
    combined_df = pd.DataFrame(index=stat_names, columns=anomaly_columns)
    for row in stat_names:
        for col in anomaly_columns:
            val_df = df.at[row, col]
            val_df1 = df1.at[row, col]
            if pd.notnull(val_df) and pd.notnull(val_df1):
                combined_df.at[row, col] = f"{int(val_df)} ({int(val_df1)})"
            elif pd.notnull(val_df):
                combined_df.at[row, col] = f"{int(val_df)}"
            elif pd.notnull(val_df1):
                combined_df.at[row, col] = f"({int(val_df1)})"
            else:
                combined_df.at[row, col] = ""
    
    # -----------------------------
    # Display the Combined Table as HTML and force MathJax typesetting
    # -----------------------------
    html_table = combined_df.to_html(escape=False)
    display(HTML(html_table))
    display(Javascript("MathJax.Hub.Queue(['Typeset', MathJax.Hub]);"))
    
    return combined_df


