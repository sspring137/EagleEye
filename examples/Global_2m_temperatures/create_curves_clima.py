#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functions to create lon x start-day-offset matrices
from the results dictionary, and to return eight DataFrames (one per season/time period 
and cluster type: OVER and UNDER).
"""

import pickle
import pandas as pd

def get_lon_sd_matrices(results):
    """
    Given the results dictionary for one season/time period, returns two DataFrames:
      - df_over: rows are center longitudes and columns are start day offsets for OVER counts.
      - df_under: rows are center longitudes and columns are start day offsets for UNDER counts.
    
    The counts are computed as the sum over clusters of the length of the "IE_extra" list.
    
    Parameters
    ----------
    results : dict
        Expected structure:
        
            results[center_lon][start_day_offset] is a dict with keys:
                'OVER_clusters': dict of clusters,
                'UNDER_clusters': dict of clusters.
                
        Each cluster is itself a dict that should contain the key "IE_extra" whose value is list-like.
    
    Returns
    -------
    df_over : pd.DataFrame
        DataFrame of OVER counts with center longitudes as index and start day offsets as columns.
    df_under : pd.DataFrame
        DataFrame of UNDER counts with center longitudes as index and start day offsets as columns.
    """
    # Sort center longitudes (as strings) and convert them to float
    lons = sorted(results.keys(), key=lambda x: float(x))
    # Assume that all center longitudes have the same start day offsets.
    # (You may wish to build the union if this is not the case.)
    first_lon = lons[0]
    sds = sorted(results[first_lon].keys(), key=lambda x: float(x))
    
    # Build dictionaries to accumulate the data for each center longitude.
    data_over = {}
    data_under = {}
    
    for lon in lons:
        row_over = {}
        row_under = {}
        for sd in sds:
            clusters_over = results[lon][sd]['OVER_clusters']
            clusters_under = results[lon][sd]['UNDER_clusters']
            
            count_over = sum(len(cluster["IE_extra"])
                             for cluster in clusters_over.values() 
                             if cluster is not None and cluster.get("IE_extra") is not None)
            count_under = sum(len(cluster["IE_extra"])
                              for cluster in clusters_under.values() 
                              if cluster is not None and cluster.get("IE_extra") is not None)
            
            # Convert start day offset to float for ordering.
            row_over[float(sd)] = count_over
            row_under[float(sd)] = count_under
        
        # Use the center longitude (converted to float) as the key for each row.
        data_over[float(lon)] = row_over
        data_under[float(lon)] = row_under
    
    # Create DataFrames from the dictionaries.
    df_over = pd.DataFrame.from_dict(data_over, orient='index')
    df_under = pd.DataFrame.from_dict(data_under, orient='index')
    
    # Convert index and columns to numeric types, then sort them.
    df_over.index = pd.to_numeric(df_over.index, errors='coerce')
    df_under.index = pd.to_numeric(df_under.index, errors='coerce')
    df_over.columns = pd.to_numeric(df_over.columns, errors='coerce')
    df_under.columns = pd.to_numeric(df_under.columns, errors='coerce')
    
    df_over = df_over.sort_index().sort_index(axis=1)
    df_under = df_under.sort_index().sort_index(axis=1)
    
    return df_over, df_under

def load_all_matrices(file_info):
    """
    Given a dictionary mapping descriptive titles to pickle filenames, loads each file and
    computes the lon x start-day-offset matrices for both OVER and UNDER counts.
    
    Parameters
    ----------
    file_info : dict
        Dictionary mapping a descriptive title (e.g. "DJF: ref 51-74; test 75-98") to a filename.
    
    Returns
    -------
    matrices : dict
        Dictionary whose keys are of the form "<title>_OVER" and "<title>_UNDER", and whose
        values are the corresponding pandas DataFrames.
        In total, 8 DataFrames will be returned (4 titles × 2 cluster types).
    """
    matrices = {}
    for title, filename in file_info.items():
        print(f"Loading {filename} for {title} ...")
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        df_over, df_under = get_lon_sd_matrices(results)
        matrices[f"{title}_OVER"] = df_over
        matrices[f"{title}_UNDER"] = df_under
    return matrices

# Example usage:
if __name__ == "__main__":
    # Example file_info dictionary (adjust the filenames as needed)
    file_info = {
        'DJF: ref 51-74; test 75-98': 'Air2m_northern_DJF_results_past.pkl',
        'JJA: ref 51-74; test 75-98': 'Air2m_northern_JJA_results_past.pkl',
        'DJF: ref 51-74; test 99-22': 'Air2m_northern_DJF_results_future.pkl',
        'JJA: ref 51-74; test 99-22': 'Air2m_northern_JJA_results_future.pkl'
    }
    
    matrices = load_all_matrices(file_info)
    
    # matrices is now a dictionary with keys like:
    #   'DJF: ref 51-74; test 75-98_OVER'
    #   'DJF: ref 51-74; test 75-98_UNDER'
    #   etc.
    #
    # For example, to inspect one of the DataFrames:
    print("DJF past OVER counts:")
    print(matrices['DJF: ref 51-74; test 75-98_OVER'].head())
    
    # Save the matrices dictionary to a pickle file.
    output_file = "matrices_dict.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(matrices, f)
    
    print(f"Saved matrices dictionary to {output_file}")
    
    DJF_ref_51_74_test_75_98_OVER = matrices['DJF: ref 51-74; test 75-98_OVER']
    DJF_ref_51_74_test_75_98_UNDER = matrices['DJF: ref 51-74; test 75-98_UNDER']
    JJA_ref_51_74_test_75_98_OVER = matrices['JJA: ref 51-74; test 75-98_OVER']
    JJA_ref_51_74_test_75_98_UNDER = matrices['JJA: ref 51-74; test 75-98_UNDER']
    DJF_ref_51_74_test_99_22_OVER = matrices['DJF: ref 51-74; test 99-22_OVER']
    DJF_ref_51_74_test_99_22_UNDER = matrices['DJF: ref 51-74; test 99-22_UNDER']
    JJA_ref_51_74_test_99_22_OVER = matrices['JJA: ref 51-74; test 99-22_OVER']
    JJA_ref_51_74_test_99_22_UNDER = matrices['JJA: ref 51-74; test 99-22_UNDER']
    
    # Create a mapping of file names to matrices.
    files_to_save = {
        'DJF_ref_51_74_test_75_98_OVER.csv': DJF_ref_51_74_test_75_98_OVER,
        'DJF_ref_51_74_test_75_98_UNDER.csv': DJF_ref_51_74_test_75_98_UNDER,
        'JJA_ref_51_74_test_75_98_OVER.csv': JJA_ref_51_74_test_75_98_OVER,
        'JJA_ref_51_74_test_75_98_UNDER.csv': JJA_ref_51_74_test_75_98_UNDER,
        'DJF_ref_51_74_test_99_22_OVER.csv': DJF_ref_51_74_test_99_22_OVER,
        'DJF_ref_51_74_test_99_22_UNDER.csv': DJF_ref_51_74_test_99_22_UNDER,
        'JJA_ref_51_74_test_99_22_OVER.csv': JJA_ref_51_74_test_99_22_OVER,
        'JJA_ref_51_74_test_99_22_UNDER.csv': JJA_ref_51_74_test_99_22_UNDER
    }
    
    # Convert each matrix to a DataFrame and save it as a CSV file.
    for filename, matrix in files_to_save.items():
        # Convert the matrix to a DataFrame.
        df = pd.DataFrame(matrix)
        # Save the DataFrame to a CSV file without the index.
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")