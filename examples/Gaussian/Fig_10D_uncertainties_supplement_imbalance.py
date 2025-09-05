#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:10:56 2025

@author: sspringe
"""

"""
Created on Mon Jul 28 13:40:52 2025

@author: sspringe
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Publication style settings (Nature Comms style)
# sns.set_style("white")
sns.set_context("paper", font_scale=1.2)
# Custom plotting settings
sns.set(style="darkgrid")
plt.rcParams.update({
    'axes.titlesize': 21,
    'axes.labelsize': 17,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 17,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.color': 'silver'
})

# # File paths and labels
# file_info = [
#     ("results_10k_flower.pkl", "10 k sample"),
#     ("results_100k_flower.pkl", "100 k sample"),
# ]

file_info = [
    ("results_10k_p025.pkl", "10 k sample, p=0.25"),
    ("results_10k_p0625.pkl", "10 k sample, p=0.625"),
]

# Distributions and metric of interest
dists   = ['Torous', 'Gaussian']
metric  = 'len_Pruned'
colors  = {'Torous': 'C0', 'Gaussian': 'C1'}
markers = {'Torous': 'o',  'Gaussian': 's'}

# Prepare the figure
loc=True
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

for ax, (fname, label) in zip(axes, file_info):
    # Load results
    with open(fname, "rb") as f:
        results = pickle.load(f)

    # Get sorted contamination levels
    contamination = sorted(results['Torous'].keys())

    # Compute means & SEM for each distribution
    stats = {}
    for dist in dists:
        data = np.array([results[dist][c][metric] for c in contamination])
        means = data.mean(axis=1)
        sem   = data.std(axis=1, ddof=1) / np.sqrt(data.shape[1])
        stats[dist] = (means, sem)

    # Plot each distribution
    for dist in dists:
        mean, sem = stats[dist]
        ax.plot(contamination, mean, '-', 
                marker=markers[dist], 
                color=colors[dist], 
                label=dist,
                linewidth = 2)
        ax.fill_between(contamination, mean - sem*3, mean + sem*2, 
                        alpha=0.3, color=colors[dist])

    # Reference y=x line for repechage
    ax.plot(contamination, contamination, '--', color='grey', linewidth=1)

    # Labels & title
    ax.set_xlabel('Contamination size')
    if loc==True:
        ax.set_ylabel('Mean Pruned count')
        loc = False
    ax.set_title(f"Pruned ({label})")
    ax.legend(frameon=False, loc='upper left')
    ax.tick_params(axis='both', which='major')

# plt.tight_layout()
# plt.savefig('fig_uncertainty_ncomms.pdf', dpi=300)
# plt.show()