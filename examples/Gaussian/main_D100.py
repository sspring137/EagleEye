#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:59:15 2025

@author: sspringe
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

import seaborn as sns
import sys
module_path = '../../eagleeye'
sys.path.append(module_path)
import EagleEye
import D100
from utils_EE import compute_the_null, partitioning_function

#%% Get the patch representations


rs_bkg = range(10)
rs_test = range(100,110)
n_points = 100000

Putative_bkg = []
Pruned_bkg = []
Repechaged_bkg = []

#% EagleEye: hyperparameters setup

p       = .5

K_M     = 500

p_ext   = 1e-5

n_jobs  = 10

#% EagleEye: Get the null
import pickle
RECOMPUTE_NULL = False
if RECOMPUTE_NULL==True:
    from utils_EE import compute_the_null
    stats_null                     = compute_the_null(p=p, K_M=K_M)
    # To save
    with open('stats_null.pkl', 'wb') as f:
        pickle.dump(stats_null, f)
else:
    # To load
    with open('stats_null.pkl', 'rb') as f:
        stats_null = pickle.load(f)
# loop bkg vs bkg
for jj in range(10):
#%%
    data1, dataset1, _, _ = D100.generate_data(n_points, 0, random_state=rs_bkg[jj])
    data2, dataset2, _, _ = D100.generate_data(n_points, 0, random_state=rs_test[jj])
    # data3, dataset3, _, _ = D100.generate_data(n_points, 1, random_state=rs_test[jj])
    
    
        
    
    #%% Anomaly detection with EE
    
    
    X = dataset1
    Y = dataset2
    # Y = dataset3
    
    

    #%%
    
    #% EagleEye: Flagging & Pruning
    
    result_dict, stats_null = EagleEye.Soar(X, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, 
                                            stats_null=stats_null, result_dict_in={})
    
    #% Cluter the Putative anomalies
    from utils_EE import partitioning_function
    clusters = partitioning_function(X,Y,result_dict,p_ext=p_ext,Z=3.65 )
    
    #% Repêchage
    
    EE_book = EagleEye.Repechage(X,Y,result_dict,clusters,p_ext=p_ext)
    
    
    
    # --- NEW: sum over all available clusters ---
    all_clusters = EE_book['Y_OVER_clusters'].values()
    
    total_putative   = sum(len(c.get('Putative', []))    for c in all_clusters)
    total_pruned     = sum(len(c.get('Pruned', []))      for c in all_clusters)
    total_repechaged = sum(len(c.get('Repechaged', []))  for c in all_clusters)
    
    Putative_bkg.append(total_putative)
    Pruned_bkg.append(total_pruned)
    Repechaged_bkg.append(total_repechaged)
    
    # Putative_bkg.append(  len(EE_book['Y_OVER_clusters'][0]['Putative'])  )
    # Pruned_bkg.append(  len(EE_book['Y_OVER_clusters'][0]['Pruned'])  )
    # Repechaged_bkg.append(  len(EE_book['Y_OVER_clusters'][0]['Repechaged'])  )

    print('Done!!')
    print(str(jj))





#%%

Putative_sig = []
Pruned_sig = []
Repechaged_sig = []


# loop bkg vs bkg+signal
for jj in range(10):
    
    data1, dataset1, _, _ = D100.generate_data(n_points, 0, random_state=rs_bkg[jj])
    # data2, dataset2, _, _ = D100.generate_data(n_points, 0, random_state=rs_test[jj])
    data3, dataset3, _, _ = D100.generate_data(n_points, 1, random_state=rs_test[jj])
    
    
        
    
    #%% Anomaly detection with EE
    
    
    X = dataset1
    # Y = dataset2
    Y = dataset3
    
    

    #%%
    
    #% EagleEye: Flagging & Pruning
    
    result_dict, stats_null = EagleEye.Soar(X, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, 
                                            stats_null=stats_null, result_dict_in={})
    
    #% Cluter the Putative anomalies
    from utils_EE import partitioning_function
    clusters = partitioning_function(X,Y,result_dict,p_ext=p_ext,Z=3.65 )
    
    #% Repêchage
    
    EE_book = EagleEye.Repechage(X,Y,result_dict,clusters,p_ext=p_ext)
    
    
    # --- NEW: sum over all available clusters ---
    all_clusters = EE_book['Y_OVER_clusters'].values()
    
    total_putative   = sum(len(c.get('Putative', []))    for c in all_clusters)
    total_pruned     = sum(len(c.get('Pruned', []))      for c in all_clusters)
    total_repechaged = sum(len(c.get('Repechaged', []))  for c in all_clusters)
    
    Putative_sig.append(total_putative)
    Pruned_sig.append(total_pruned)
    Repechaged_sig.append(total_repechaged)

    print('Done!!')
    print(str(jj))
fig = plt.figure()
dimsss=[17,7]
ax = fig.add_subplot(111)
ax.scatter(dataset3[:10000, dimsss[0]],
           dataset3[:10000, dimsss[1]],
           label='Background', alpha=0.7)
ax.scatter(dataset3[-1000:, dimsss[0]],
           dataset3[-1000:, dimsss[1]],
           label='Anomaly')
ax.legend()


#%%
dict_100D = {
    'Putative_bkg': np.array(Putative_bkg),
    'Pruned_bkg': np.array(Pruned_bkg),
    'Repechaged_bkg': np.array(Repechaged_bkg),
    'Putative_sig': np.array(Putative_sig),
    'Pruned_sig': np.array(Pruned_sig),
    'Repechaged_sig': np.array(Repechaged_sig)
}
with open('dict_100D.pkl', 'wb') as f:
    pickle.dump(dict_100D, f)
    
#%%
import pandas as pd

# Assuming dict_100D is already defined...
rows = []
for key, arr in dict_100D.items():
    mean = arr.mean()
    std  = arr.std(ddof=0)   # population std; use ddof=1 for sample std
    typ, cls = key.split('_', 1)
    rows.append({
        'Type': typ.capitalize(),
        'Class': cls,
        'Mean': mean,
        'Std': std
    })

df = pd.DataFrame(rows, columns=['Type', 'Class', 'Mean', 'Std'])
print(df.to_markdown(index=False))   


#%%

import time
t = time.time()



#% EagleEye: Flagging & Pruning

result_dict, stats_null = EagleEye.Soar(X, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, 
                                        stats_null=stats_null, result_dict_in={})

#% Cluter the Putative anomalies
from utils_EE import partitioning_function
clusters = partitioning_function(X,Y,result_dict,p_ext=p_ext,Z=3.65 )

#% Repêchage

EE_book = EagleEye.Repechage(X,Y,result_dict,clusters,p_ext=p_ext)

elapsed = time.time() - t
print(elapsed)