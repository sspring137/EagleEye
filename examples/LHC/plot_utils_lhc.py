import numpy as np
def plot_third_subplot(ax,
                       Upsilon_i,
                       null_distribution,
                       Upsilon_star_plus,
                       Upsilon_set_equalized,
                       Upsilon_set_repechage,
                       Upsilon_set_pruned,
                       n_bins=100,
                       legend = True,
                       xlog=False):
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
                       np.log10(np.max(Upsilon_i)),
                       n_bins)
    if xlog==False:
        bins = np.linspace(np.min(Upsilon_i),
                        np.max(Upsilon_i),
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
    if xlog==True:
        ax.set_xscale('log')
    ax.set_yscale('log')

    # Increase tick thickness.
    ax.tick_params(axis='both', which='major', length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=1.5)

    # Plot Original dataset and anomalies as bars with white edges.
    ax.bar(bin_centers, norm_Upsilon_i, width=bin_widths, color='silver',
           alpha=0.6, edgecolor='white', linewidth=2,
           label=r'$\mathcal{Y}^+$', align='center')
    ax.bar(bin_centers, norm_Upsilon_set_repechage, width=bin_widths, color='limegreen',
           alpha=0.6, edgecolor='white', linewidth=2,
           label=r'$\mathcal{Y}_{\alpha}^{anom}$', align='center')

    ax.bar(bin_centers, norm_Upsilon_set_pruned, width=bin_widths, color='darkgreen',
           alpha=0.6, edgecolor='white', linewidth=2,
           label=r'$\hat{\mathcal{Y}}^+$', align='center')



    # Plot the equalized dataset and null distribution as lines.
    ax.plot(bin_centers, norm_null_distribution, marker='.', color='black',
            label=r'Null distribution')

    ax.bar(bin_centers, norm_Upsilon_set_equalized, width=bin_widths, color='royalblue',
           alpha=0.6, edgecolor='white', linewidth=2,
           label=r'$\mathcal{Y}^{eq}$', align='center')

    # Add a vertical dashed red line at the critical threshold.
    ax.axvline(x=Upsilon_star_plus, color='red', linestyle='--', linewidth=2,
               label=r'$\mathbf{\Upsilon}_+^{*}$')

    # Set the x and y limits.
    ax.set_xlim([1.5, max(Upsilon_i)])
    ax.set_ylim([1e-5, 1])

    # Labeling and legend.
    ax.set_xlabel(r'$\mathbf{\Upsilon}_i^+$')

    handles, labels = ax.get_legend_handles_labels()


    desired_order = [0, 1, 2, 4, 3, 5 ]

    # Reorder the handles and labels according to the desired order.
    ordered_handles = [handles[i] for i in desired_order]
    ordered_labels = [labels[i] for i in desired_order]

    # Now create the legend with the new ordering.
    if legend:
        ax.legend(ordered_handles, ordered_labels,fontsize=10)
    #ax.legend()


def MAS2(dict,repechage_EE_book,Upsilon_star_plus=14):
    over_clusters  = repechage_EE_book["Y_OVER_clusters"] # Need to generate as a function of Upsilon_star
    l   = dict['stats']['lables_mix']
    s   = dict['Upsilon_i_Y'][(dict['Upsilon_i_Y']>Upsilon_star_plus) & (l==1)]
    b   = dict['Upsilon_i_Y'][(dict['Upsilon_i_Y']>Upsilon_star_plus) & (l==0)]

    s = np.concatenate([dict['Upsilon_i_Y'][(np.isin(np.arange(len(dict['Upsilon_i_Y'])), over_clusters[idx]['Repechaged']) & (l == 1))] for idx in list(over_clusters.keys())])
    b = np.concatenate([dict['Upsilon_i_Y'][(np.isin(np.arange(len(dict['Upsilon_i_Y'])), over_clusters[idx]['Repechaged']) & (l == 0))] for idx in list(over_clusters.keys())])
    S_B = len(s)/np.sqrt(len(b))
    S_B = {'Total' : S_B}
    SoUpsilons     =  np.concatenate([dict['Upsilon_i_Y'][over_clusters[idx]['Repechaged']] for idx in list(over_clusters.keys())])
    Upsilon_xi     = {'Total' : min(SoUpsilons)}
    # Now get per mode info
    for idx in list(over_clusters.keys()):
        lenSo          =  len(over_clusters[idx]['Repechaged'])
        if lenSo < 5:
            continue
        Upsilon_xi[idx] = min(dict['Upsilon_i_Y'][over_clusters[idx]['Repechaged']])
        s = dict['Upsilon_i_Y'][(dict['Upsilon_i_Y'] > Upsilon_xi[idx]) & (l == 1) & np.isin(np.arange(len(l)), over_clusters[idx]['Repechaged'])]
        b = dict['Upsilon_i_Y'][(dict['Upsilon_i_Y'] > Upsilon_xi[idx]) & (l == 0) & np.isin(np.arange(len(l)), over_clusters[idx]['Repechaged'])]
        S_B[idx] = len(s)/np.sqrt(len(b))
    return S_B

def percentage_of_anomaly(anomaly_size):
    return 100 * anomaly_size / 500000
