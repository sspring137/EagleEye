#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:34:23 2025

@author: sspringe
"""
import utils_sistematics

# ============================================================
# Example usage
# ============================================================

#%% Stage 1
X1, X2, Y = utils_sistematics.stage1_make_data(cont=1000)

#%% Stage 2
results, p_values = utils_sistematics.stage2_run_soar(X1, X2, Y, K_M=500, p_ext=1e-5, n_jobs=10)

# (Later / in another script you could just load from disk)
# with open("eagleeye_results.pkl", "rb") as f:
#     payload   = pickle.load(f)
#     results   = payload["results"]
#     p_values  = payload["p_values"]

#%% Stage 3
PostProcessing = utils_sistematics.stage3_analyze(X1, X2, Y, results, p_values, Z=3.65, K_M=500, p_ext=1e-5, make_plots=True)
