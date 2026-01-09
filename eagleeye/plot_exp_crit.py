#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:02:55 2025

@author: sspringe
"""
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from  critical_proportion import expected_max_binomial_exact_scipy, get_trashold

# Parameters
ps = [0.3, 0.5, 0.9]
Numb_rep = int(1e6)
KKK = np.arange(1, 10000, 10)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Compute and plot yyy = E[max]/k for each p
loc=True
for ax, p in zip(axes, ps):
    yyy = [expected_max_binomial_exact_scipy(k, p, Numb_rep) / k for k in KKK]
    ax.plot(KKK, yyy, label=r'$\mathbb{E}[X^\dag]/K$', linewidth=2)
    ax.axhline(p, color='gray', linestyle='--', label=f'$p={p}$', linewidth=2)
    ax.set_title(f'$p = {p}$', fontsize=18)
    ax.set_xlabel('Neighborhood rank $K$', fontsize=16)
    if loc:
        ax.set_ylabel(r'$\mathbb{E}[X^\dag]/K$', fontsize=16)
        loc=False
    ax.set_ylim([-0.1*( 1-p )+p, 1.01])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)

plt.show()

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Compute and plot yyy = E[max]/k for each p
loc=True
for ax, p in zip(axes, ps):
    yyy = [get_trashold(k, p, Numb_rep)  for k in KKK]
    ax.plot(KKK, yyy, label=r'$Anomaly \ score $', linewidth=2)
    # ax.axhline(p, color='gray', linestyle='--', label=f'$p={p}$', linewidth=2)
    ax.set_title(f'$p = {p}$', fontsize=18)
    ax.set_xlabel('Neigh. rank $K$', fontsize=16)
    if loc:
        ax.set_ylabel(r'$\mathbf{\Upsilon}_+(\mathbb{E}[X^\dag(K)])$', fontsize=16)
        loc=False
    ax.set_ylim([10,17])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)

plt.show()

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Compute and plot yyy = E[max]/k for each p
loc=True
k = 1000
Nr=[10**i for i in range(3, 17)]
for ax, p in zip(axes, ps):
    yyy = [get_trashold(k, p, Numb_rep)  for Numb_rep in Nr]
    ax.plot(Nr, yyy, label=r'$Crit. trash. $', linewidth=2)
    # ax.axhline(p, color='gray', linestyle='--', label=f'$p={p}$', linewidth=2)
    ax.set_title(f'$p = {p}$', fontsize=18)
    ax.set_xlabel('Max Neigh. rank $K_M$', fontsize=16)
    if loc:
        ax.set_ylabel(r'Crit. trash.', fontsize=16)
        loc=False
    # ax.set_ylim([10,17])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)

plt.show()
