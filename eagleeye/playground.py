#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:13:10 2025

@author: sspringe
"""


import numpy as np
import matplotlib.pyplot as plt
import EagleEye
# import time

#%% Generate the data
cont = 1000
X= np.random.randn(150000,3)
# X= np.random.randn(50000-cont,3)
# Y= np.random.randn(50000,3)
Y= np.random.randn(50000-cont,3)
# X = np.concatenate((X, -1- np.random.randn(cont,3)/10)).astype(float)
Y = np.concatenate((Y, 1.+ np.random.randn(cont,3)/10)).astype(float) 


#%% EagleEye hyperparameters

p       = len(Y)/(len(Y)+len(X))

K_M     = 500

p_ext   = 1e-5

n_jobs  = 10

#%% Get the null

from utils_EE import compute_the_null
stats_null                     = compute_the_null(p=p, K_M=K_M)

#%% Eagle Soar!

# t = time.time()
result_dict, stats_null = EagleEye.Soar(X, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={})
# elapsed17alt = time.time() - t
# print(f'Elapsed time: {elapsed17alt} seconds')

#%% Cluter the Putative anomalies
from utils_EE import partitioning_function
clusters = partitioning_function(X,Y,result_dict,p_ext=p_ext,Z=2.65 )

#%% Repêchage

EE_book = EagleEye.Repechage(X,Y,result_dict,clusters,p_ext=p_ext)


#%% Visualizations 

#%%
Putative   = EE_book['Y_OVER_clusters'][0]['Putative']
Pruned     = EE_book['Y_OVER_clusters'][0]['Pruned']
Repechaged = EE_book['Y_OVER_clusters'][0]['Repechaged']
Background = EE_book['Y_OVER_clusters'][0]['Background']

    
#%%

fig = plt.figure()
# Plotting the scatterplot
plt.scatter(Y[:-cont,0], Y[:-cont,1],marker='.', s=1, c='silver', alpha=0.3)
if any(Putative):
    plt.scatter(Y[Putative,0], Y[Putative,1],marker='.', s=1, c='red', alpha=0.7)
if any(Repechaged):
    plt.scatter(Y[Repechaged,0], Y[Repechaged,1],marker='.', s=1, c='limegreen', alpha=0.7)
if any(Pruned):
    plt.scatter(Y[Pruned,0], Y[Pruned,1],marker='.', s=1, c='darkgreen', alpha=0.7)
if any(Background):
    plt.scatter(X[Background,0], X[Background,1],marker='.', s=1, c='gold', alpha=0.7)

# Displaying the plot
plt.show()



#%%

Putative   = EE_book['X_OVER_clusters'][0]['Putative']
Pruned     = EE_book['X_OVER_clusters'][0]['Pruned']
Repechaged = EE_book['X_OVER_clusters'][0]['Repechaged']
Background = EE_book['X_OVER_clusters'][0]['Background']
#%%

# fig = plt.figure()
# Plotting the scatterplot
# plt.scatter(X[:-cont,0], X[:-cont,1],marker='.', s=1, c='silver', alpha=0.3)
if any(Putative):
    plt.scatter(X[Putative,0], X[Putative,1],marker='.', s=1, c='red', alpha=0.7)
if any(Repechaged):
    plt.scatter(X[Repechaged,0], X[Repechaged,1],marker='.', s=1, c='limegreen', alpha=0.7)
if any(Pruned):
    plt.scatter(X[Pruned,0], X[Pruned,1],marker='.', s=1, c='darkgreen', alpha=0.7)
if any(Background):
    plt.scatter(Y[Background,0], Y[Background,1],marker='.', s=1, c='gold', alpha=0.7)
# Displaying the plot
plt.legend(['Non anom','Putative', 'Repechaged', 'Pruned', 'Background'], markerscale=5, fontsize=10)
plt.show()
