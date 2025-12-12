#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 13:13:11 2025

@author: sspringe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle

import EagleEye
from EagleEye import PValueCalculator
from utils_EE import compute_the_null, partitioning_function


# ============================================================
# Stage 1 – load or create X1, X2, Y
# ============================================================

def stage1_make_data(cont: int = 1000, seed: int | None = None):
    """
    Create example datasets X1, X2, Y.
    Replace this with your real data loader when needed.
    """
    if seed is not None:
        np.random.seed(seed)

    X1 = np.random.randn(50000 - cont, 3)
    X2 = np.random.randn(50000 - cont, 3)
    Y  = np.random.randn(50000 - 2 * cont, 3)

    # Add structured components / anomalies
    X2 = np.concatenate((X2, -1.3 - np.random.randn(cont, 3) / 10)).astype(float)
    Y  = np.concatenate((Y,  1.0 + np.random.randn(cont, 3) / 10)).astype(float)
    Y  = np.concatenate((Y, -1.0 - np.random.randn(cont, 3) / 10)).astype(float)
    X1 = np.concatenate((X1,  1.03 + np.random.randn(cont, 3) / 10)).astype(float)

    return X1, X2, Y


# ============================================================
# Stage 2 – run EagleEye.Soar on all pairs and save results
# ============================================================

def stage2_run_soar(
    X1: np.ndarray,
    X2: np.ndarray,
    Y:  np.ndarray,
    K_M: int = 500,
    p_ext: float = 1e-5,
    n_jobs: int = 10,
    results_path: str = "eagleeye_results.pkl",
):
    """
    For each of (X1,X2), (X2,X1), (X1,Y), (X2,Y):
        - compute p
        - compute stats_null
        - run EagleEye.Soar

    Saves:
        {'results': {pair_name: result_dict}, 'p_values': {pair_name: p}}
    """
    pairs = {
        "X1_X2": (X1, X2),
        "X2_X1": (X2, X1),
        "X1_Y":  (X1, Y),
        "X2_Y":  (X2, Y),
    }

    results   = {}
    p_values  = {}

    for name, (A, B) in pairs.items():
        p = len(B) / (len(A) + len(B))          # as requested: p = len(second) / total
        p_values[name] = p

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
        # stats_null = compute_the_null(p=p, K_M=K_M)
        
        
        result_dict, _ = EagleEye.Soar(
            A, B,
            K_M=K_M,
            p_ext=p_ext,
            n_jobs=n_jobs,
            stats_null=stats_null,
            result_dict_in={}
        )
        results[name] = result_dict

    payload = {"results": results, "p_values": p_values}
    with open(results_path, "wb") as f:
        pickle.dump(payload, f)

    return results, p_values


# ============================================================
# Stage 3 – clustering, repêchage, optional viz + fractions
# ============================================================

def _plot_contour(pts: np.ndarray, cmap: str):
    kde = gaussian_kde(pts)
    xmin, xmax = pts[0].min(), pts[0].max()
    ymin, ymax = pts[1].min(), pts[1].max()

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 300),
        np.linspace(ymin, ymax, 300),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=17, linewidths=1, cmap=cmap)


def _plot_anom_vs_sistematics(
    Y,
    X1,
    X2,
    EE_book_sistematicsX1X2,
    EE_book_sistematicsX2X1,
    EE_book,
    title: str,
):
    plt.figure()
    plt.title(title)

    # Background: all Y (subsample for speed)
    pts_bg = Y[:-2000, :2][::5].T
    _plot_contour(pts_bg, cmap="Greys")

    # Systematics from X1
    syst_X1_idx = EE_book_sistematicsX2X1["Y_OVER_clusters"][0]["Repechaged"]
    pts_X1 = X1[syst_X1_idx, :2].T
    _plot_contour(pts_X1, cmap="Blues")

    # Systematics from X2
    syst_X2_idx = EE_book_sistematicsX1X2["Y_OVER_clusters"][0]["Repechaged"]
    pts_X2 = X2[syst_X2_idx, :2].T
    _plot_contour(pts_X2, cmap="Purples")

    # Anomalies in Y
    for cluster in EE_book["Y_OVER_clusters"].values():
        anom_idx = cluster["Repechaged"]
        pts_anom = Y[anom_idx, :2].T
        _plot_contour(pts_anom, cmap="Reds")

    plt.legend(["Background", "Systematics in X1", "Systematics in X2", "Anomalies"])


def stage3_analyze(
    X1: np.ndarray,
    X2: np.ndarray,
    Y:  np.ndarray,
    results: dict,
    p_values: dict,
    Z: float = 2.65,
    K_M: int = 500,
    p_ext: float = 1e-5,
    make_plots: bool = True,
):
    """
    Stage 3:
        - cluster putative anomalies
        - run Repêchage
        - optional visualization
        - compute intersection fractions with systematics

    Parameters
    ----------
    X1, X2, Y : datasets
    results   : dict from stage 2 (pair_name -> result_dict)
    p_values  : dict from stage 2 (pair_name -> p)
    Z         : Z-score threshold for clustering
    """
    # Unpack results for readability
    result_sist_X1X2 = results["X1_X2"]
    result_sist_X2X1 = results["X2_X1"]
    result_X1Y       = results["X1_Y"]
    result_X2Y       = results["X2_Y"]

    # --- Clustering ---
    clusters_sist_X1X2 = partitioning_function(X1, X2, result_sist_X1X2, p_ext=p_ext, Z=Z)
    clusters_sist_X2X1 = partitioning_function(X2, X1, result_sist_X2X1, p_ext=p_ext, Z=Z)
    clusters_X1Y       = partitioning_function(X1, Y,  result_X1Y,       p_ext=p_ext, Z=Z)
    clusters_X2Y       = partitioning_function(X2, Y,  result_X2Y,       p_ext=p_ext, Z=Z)

    # --- Repêchage ---
    EE_book_sist_X1X2 = EagleEye.Repechage(X1, X2, result_sist_X1X2, clusters_sist_X1X2, p_ext=p_ext)
    EE_book_sist_X2X1 = EagleEye.Repechage(X2, X1, result_sist_X2X1, clusters_sist_X2X1, p_ext=p_ext)
    EE_book_X1Y       = EagleEye.Repechage(X1, Y,  result_X1Y,       clusters_X1Y,       p_ext=p_ext)
    EE_book_X2Y       = EagleEye.Repechage(X2, Y,  result_X2Y,       clusters_X2Y,       p_ext=p_ext)

    # --- Optional visualization ---
    if make_plots:
        _plot_anom_vs_sistematics(
            Y, X1, X2,
            EE_book_sist_X1X2,
            EE_book_sist_X2X1,
            EE_book_X1Y,
            title="(X1 → Y)",
        )
        _plot_anom_vs_sistematics(
            Y, X1, X2,
            EE_book_sist_X1X2,
            EE_book_sist_X2X1,
            EE_book_X2Y,
            title="(X2 → Y)",
        )

    # --- Fractions of systematics inside anomaly clusters ---
    KSTAR_RANGE = range(20, K_M)
    n_ref = X1.shape[0]   # reference length (used in binary_seq construction)

    p_sist_X1X2 = p_values["X1_X2"]
    p_sist_X2X1 = p_values["X2_X1"]

    Fractions_Sistematics_X1 = []
    for EE_books_Y in [EE_book_X1Y, EE_book_X2Y]:
        for cluster in EE_books_Y["Y_OVER_clusters"].values():
            anom_idx = cluster["Repechaged"]

            indices = result_sist_X2X1["Knn_model"].kneighbors(Y[anom_idx, :])[1]
            binary_seq = (indices > n_ref).astype(int)
            binary_seq = binary_seq[:, :K_M]
            binary_seq[:n_ref, 0] = 1  # injection 1-by-1

            p_val_info = PValueCalculator(binary_seq, KSTAR_RANGE, p=p_sist_X2X1)
            Upsilon_i_inj = p_val_info.min_pval_plus

            temp = indices[:, :K_M][0, :] - n_ref
            overlap_idx = [
                idx for idx in temp
                if idx in EE_book_sist_X2X1["Y_OVER_clusters"][0]["Repechaged"]
            ]

            if len(overlap_idx) > 1:
                Upsilon_alpha_plus = EE_book_sist_X2X1["Y_OVER_clusters"][0]["Upsilon_alpha_plus"]
            else:
                Upsilon_alpha_plus = np.inf

            Fractions_Sistematics_X1.append(
                np.sum(Upsilon_i_inj > Upsilon_alpha_plus) / len(Upsilon_i_inj)
            )

    Fractions_Sistematics_X2 = []
    for EE_books_Y in [EE_book_X1Y, EE_book_X2Y]:
        for cluster in EE_books_Y["Y_OVER_clusters"].values():
            anom_idx = cluster["Repechaged"]

            indices = result_sist_X1X2["Knn_model"].kneighbors(Y[anom_idx, :])[1]
            binary_seq = (indices > n_ref).astype(int)
            binary_seq = binary_seq[:, :K_M]
            binary_seq[:n_ref, 0] = 1

            p_val_info = PValueCalculator(binary_seq, KSTAR_RANGE, p=p_sist_X1X2)
            Upsilon_i_inj = p_val_info.min_pval_plus

            temp = indices[:, :K_M][0, :] - n_ref
            overlap_idx = [
                idx for idx in temp
                if idx in EE_book_sist_X1X2["Y_OVER_clusters"][0]["Repechaged"]
            ]

            if len(overlap_idx) > 1:
                Upsilon_alpha_plus = EE_book_sist_X1X2["Y_OVER_clusters"][0]["Upsilon_alpha_plus"]
            else:
                Upsilon_alpha_plus = np.inf

            Fractions_Sistematics_X2.append(
                np.sum(Upsilon_i_inj > Upsilon_alpha_plus) / len(Upsilon_i_inj)
            )

    print("Fraction of X1 Systematics:")
    print("----------------------------------------------------------------------")
    print(f"C1(X1 -> Y): {Fractions_Sistematics_X1[0]}")
    print(f"C2(X1 -> Y): {Fractions_Sistematics_X1[1]}")
    print(f"C1(X2 -> Y): {Fractions_Sistematics_X1[2]}")
    print(f"C2(X2 -> Y): {Fractions_Sistematics_X1[3]}")

    print("\nFraction of X2 Systematics:")
    print("----------------------------------------------------------------------")
    print(f"C1(X1 -> Y): {Fractions_Sistematics_X2[0]}")
    print(f"C2(X1 -> Y): {Fractions_Sistematics_X2[1]}")
    print(f"C1(X2 -> Y): {Fractions_Sistematics_X2[2]}")
    print(f"C2(X2 -> Y): {Fractions_Sistematics_X2[3]}")

    return {
        "Fractions_Sistematics_X1": Fractions_Sistematics_X1,
        "Fractions_Sistematics_X2": Fractions_Sistematics_X2,
        "EE_books": {
            "sist_X1X2": EE_book_sist_X1X2,
            "sist_X2X1": EE_book_sist_X2X1,
            "X1Y":       EE_book_X1Y,
            "X2Y":       EE_book_X2Y,
        },
    }

