#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:11:09 2024

@author: sspringe
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
# from netCDF4 import Dataset
import matplotlib.cm as cm 
from mpl_toolkits.basemap import Basemap
import Blocking_labels 
import NCEP_500hPa_functions
from IPython.display import display
import pickle
import os
import itertools
import operator
from scipy.signal.windows import gaussian
from sklearn.cluster import kmeans_plusplus
from typing import Tuple
from scipy.linalg import eig
from scipy.linalg import lu
from scipy.linalg import solve
# apparently this avoid an error due to incompatibility between scipy and torch
_, _,_ =eig( np.eye(2), left = True )
import torch

class Data:
     def __init__(self, current_script_path, data_name, grid_discretization, data_extension, scale, lati, longi, window, season, filters, n_microstates):
        """Initialize the Data object with all necessary attributes."""
        self.current_script_path = current_script_path
        self.data_name = data_name
        self.data_extension = data_extension
        self.grid_discretization = grid_discretization
        self.scale = scale
        self.lati = lati
        self.longi = longi
        self.window = window
        self.season = season
        self.filters = filters
        self.n_microstates = n_microstates

     
     def show(self):
        """Print all the attributes of the Data object."""
        attrs = vars(self)
        print('; \n'.join(f"{key}: {value}" for key, value in attrs.items()))
         
     def load_data(self):
        """Load data from a numpy file, handle missing files and calculate percentiles."""
        try:
            
            data_path = os.path.join(self.current_script_path, 'DATA', self.data_name + self.data_extension)
            self.AIR2M = np.load(data_path)
            self.AIR2M_filtered = self.AIR2M.copy() 
            self.p1 = np.percentile(self.AIR2M, 1)
            self.p97 = np.percentile(self.AIR2M, 97)
        except FileNotFoundError:
            print(f"Error: The file {data_path} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
     def detrend_and_deseasonalize_data(self):
        """Detrend and deseasonalize the data."""
        
        season_indices = {
            "Winter": np.concatenate((np.arange(60), np.arange(335, 365))),
            "Summer": np.arange(150, 240)
        }
        try:
            season_length = len(season_indices[self.season])
            
            if (self.season == "Summer"):
            
                # Calculate the number of chunks of 90 values
                num_chunks = self.AIR2M.shape[0] // 90
    
                # Reshape the data into chunks of size 30 along the first axis
                data_chunks = self.AIR2M[:num_chunks * 90,:,:].reshape(num_chunks, 90, *self.AIR2M.shape[1:])
    
                # Compute the mean along the second axis (30 values) to obtain averaged values
                averaged_values = np.mean(data_chunks, axis=1)
    
                # Repeat each averaged value 90 times
                trend_tbr = np.repeat(averaged_values, 90, axis=0)
            
            elif (self.season == "Winter"):
                # Calculate the number of chunks of 90 values
                num_chunks = self.AIR2M.shape[0] // 90
                # Calculate the number of complete chunks of 90 values (excluding the first and last chunks)
                num_complete_chunks = num_chunks - 1
                
                # Calculate the number of values in the first chunk and the last chunk
                length_first_chunk = 60
                length_last_chunk = 30
                
                # Reshape the data into chunks of size 90 along the first axis for the complete chunks
                complete_chunks_data = self.AIR2M[length_first_chunk:-length_last_chunk,:,:]
                complete_chunks = complete_chunks_data.reshape(num_complete_chunks, 90, *self.AIR2M.shape[1:])
                
                # Compute the mean along the second axis (90 values) to obtain averaged values for the complete chunks
                complete_chunks_averages = np.mean(complete_chunks, axis=1)
                
                # Repeat each averaged value 90 times for the complete chunks
                complete_chunks_trend = np.repeat(complete_chunks_averages, 90, axis=0)
                
                # Extract the data for the first and last chunks
                first_chunk_data = self.AIR2M[:length_first_chunk,:,:]
                last_chunk_data = self.AIR2M[-length_last_chunk:,:,:]
                
                # Compute the mean for the first and last chunks
                first_chunk_average = np.mean(first_chunk_data, axis=0)[np.newaxis,:,:]
                last_chunk_average = np.mean(last_chunk_data, axis=0)[np.newaxis,:,:]
                
                # Repeat the averaged values for the first and last chunks
                first_chunk_trend = np.repeat(first_chunk_average, length_first_chunk, axis=0)
                last_chunk_trend = np.repeat(last_chunk_average, length_last_chunk, axis=0)
                
                # Concatenate the trend values for the first chunk, complete chunks, and last chunk
                trend_tbr = np.concatenate((first_chunk_trend, complete_chunks_trend, last_chunk_trend), axis=0)
                
            temporary = self.AIR2M - trend_tbr
            
            # Reshape the temporary variable into chunks of size 90 along the first axis
            temp_chunks = temporary.reshape(-1, 90, *temporary.shape[1:])

            # Compute the mean along the second axis (90 values) to obtain averaged values
            temp_chunks1 = np.mean(temp_chunks, axis=0)

            # Repeat each averaged value num_chunks times
            season_tbr = np.tile(temp_chunks1, (num_chunks,1,1))
            
            self.AIR2M_filtered = self.AIR2M - trend_tbr - season_tbr
            
        except KeyError:
            print(f"Error: Season {self.season} is not defined.")

     def get_blocked_days(self):
        # Compute blocking events using the TM index method
        self.blocked_days = self.compute_TM_blocking(self.AIR2M, self.AIR2M.shape[0])

     def compute_TM_blocking(self, psi, time):
        # Prepare an array for holding blocking results
        Blocking = np.zeros((time, psi.shape[2]), dtype=bool)
        
        # Compute blocking using the Tibaldi-Molteni index for each time step
        for i in range(time):
            Blocking[i, :], _, _ = self.TM_indexes2(psi, i)
        
        # Calculate the length of blocking events across time
        return self.calculate_blocking_lengths(Blocking)

     def TM_indexes2(self, psi, t):
        # Computes the Tibaldi-Molteni index to determine atmospheric blocking
        GHGN = np.zeros((psi.shape[2], 5))  # Northward gradient
        GHGS = np.zeros((psi.shape[2], 5))  # Southward gradient
        
        latitudes = np.linspace(0, 90, psi.shape[1])
        
        # Define indices for critical latitudes based on expected TM analysis
        critical_indices = {
            'north': np.argmin(np.abs(latitudes - 80)),
            'central': np.argmin(np.abs(latitudes - 60)),
            'south': np.argmin(np.abs(latitudes - 40)),
        }
        
        for i in range(-2, 3):
            Z_N = psi[t, critical_indices['north'] + i, :]
            Z_0 = psi[t, critical_indices['central'] + i, :]
            Z_S = psi[t, critical_indices['south'] + i, :]
            
            GHGN[:, i] = Z_0 - Z_N
            GHGS[:, i] = Z_0 - Z_S
        
        GHG = (GHGN > 150) & (GHGS > 0)  # Evaluate blocking condition
        
        return GHG.sum(axis=1) > 0, GHGN.ravel(), GHGS.ravel()

     def calculate_blocking_lengths( self, Blocking ):

        # Calculate the consecutive length of blocking days
        Index_list = []
        Blocked_days_length = np.zeros(Blocking.shape)
        # group the block based on their index
        for ind in range( Blocking.shape[1] ):
            auxy = [[i for i,value in it] for key,it in \
                    itertools.groupby(enumerate( Blocking[ :, ind ] ), \
                                      key=operator.itemgetter(1)) if key != 0]
            # define the length in the matrix form    
            for jnd in range( len( auxy ) ):
                Blocked_days_length[ auxy[ jnd ] , ind ] = len( auxy[ jnd ] )
                
            Index_list.append( auxy )
            
        return Blocked_days_length
    
     def get_microstates(self) -> None:
        self.apply_compatibility_fix()
        self.scale_sqrt, grid_filter = self.calculate_distance_weights()

        self.data_tbc = self.reshape_data_for_clustering()
        self.init_partition, self.centroids = self.weighted_kmeans(self.data_tbc)
        
        # del self.AIR2M_filtered  # Clean up the large data structure

     def apply_compatibility_fix(self) -> None:
        """Apply a compatibility fix between scipy and torch."""
        _, _, _ = eig(np.eye(2), left=True)

     def calculate_distance_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate weighted filters based on latitude and longitude."""
        weights1 = np.cos(np.deg2rad(np.array(self.lati) * self.scale)) if 'Lati_area' in self.filters else np.ones(len(self.lati))
        window = gaussian(len(self.longi), std=len(self.longi)/2) if 'Longi_Gaussian' in self.filters else np.ones(len(self.longi))
        
        grid_filter = np.zeros( ( self.grid_discretization[0], len(self.longi) ) )
       
        final_weights = []
        for jj in range(window.shape[0]):
            final_weights.extend( weights1*window[jj] )
            grid_filter[self.lati,jj] = ( weights1*window[jj] )
  
        return np.sqrt(np.array( final_weights )),grid_filter    

     def reshape_data_for_clustering(self) -> np.ndarray:
        """Reshape data for clustering."""
        data_tbc = self.AIR2M[:, self.lati, :][:, :, self.longi]
        return data_tbc.reshape((data_tbc.shape[0], np.prod(data_tbc.shape[1:])), order='F').copy(order='C')

     def weighted_kmeans(self, data_tbc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform weighted K-means clustering."""
        scaled_data = data_tbc * self.scale_sqrt
        init_centroids, _ = kmeans_plusplus(scaled_data, n_clusters=self.n_microstates, n_local_trials=100000)
        centroids = init_centroids.copy()
        
        indata_Km_torch = torch.tensor(scaled_data, dtype=torch.float32)
        for _ in range(700):  # Use a constant for the maximum number of iterations
            centroids_torch = torch.tensor(centroids, dtype=torch.float32)
            distances = torch.cdist(indata_Km_torch, centroids_torch, p=2.0)
            cluster_assignment = torch.argmin(distances, axis=1).numpy()

            new_centroids = np.array([data_tbc[cluster_assignment == i].mean(axis=0) for i in range(self.n_microstates)]) * self.scale_sqrt
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return cluster_assignment, centroids
    
     def get_perc_blocked_days_per_micro(self):
        """
        Calculate the percentage of blocked days per microstate across all longitudes.
        The percentage is calculated only for days where the blocking is more than 2 days.
        """
        # Calculate the percentage of blocked days for each day across relevant longitudes
        percentage_blocked = (self.blocked_days[:, self.longi] > 2).sum(axis=1) / len(self.longi)
    
        # Calculate the mean percentage of blocked days for each microstate
        blocked_days_cluster = [percentage_blocked[self.init_partition == jj].mean() for jj in range(self.n_microstates)]
    
        self.blocked_days_per_micro = np.array(blocked_days_cluster)
        
     def transition_matrix(self, tau=1, do_plot=False):
        """
        Generate a transition matrix from time-series data of state indices using specified time lag.
        Skips transitions from the 60th day and then every 90th day thereafter.
        """
        t = np.array(self.init_partition)
        total_indices = len(t) - tau
        M = np.zeros((self.n_microstates, self.n_microstates))

        # Days to skip: starting from the 60th day and then every 90th day
        skip_days = set(range(59, total_indices, 90))

        # Constructing the transition matrix without including skipped days
        for i in range(total_indices):
            if i not in skip_days:
                M[t[i], t[i + tau]] += 1

        # Normalize the matrix to create a probability matrix
        row_sums = M.sum(axis=1)
        M = np.divide(M, row_sums[:, np.newaxis], out=np.zeros_like(M), where=row_sums[:, np.newaxis] != 0)

        # Plotting the transition matrix if requested
        if do_plot:
            plt.figure()
            plt.imshow(M, interpolation='nearest', cmap='viridis')
            plt.title("Transition Matrix")
            plt.colorbar()
            plt.show()

        return M

     def eigenvalue_decomposition(self):
        """
        Perform eigen decomposition of the matrix M, normalize eigenvectors,
        verify the decomposition, and adjust the basis if necessary.
        Returns eigenvalues and both left and right eigenvectors.
        """
        # Calculate eigenvalues and eigenvectors
        lam, vl, vr = eig(self.M, left=True)
        idx = np.abs(lam).argsort()[::-1]
        lam = lam[idx]
        vl = vl[:, idx].conj()  # Take the conjugate of the left eigenvectors
        vr = vr[:, idx]  # Right eigenvectors

        # Normalize the primary eigenvector
        vr[:, 0] = vr[:, 0] / vr[0, 0]
        normalization_constant = np.sum(vr[:, 0] * vl[:, 0]).real
        vl[:, 0] = vl[:, 0] / normalization_constant

        # Verify the reconstruction of matrix M
        reconstructed_M = np.zeros_like(self.M, dtype=np.complex_)
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[0]):
                reconstructed_M[i, j] = np.sum(vr[:, i] * vl[:, j].T)

        # Perform LU decomposition to adjust bases
        P, L, U = lu(reconstructed_M)
        vl_primary = np.linalg.inv(L) @ vl
        vr_primary = vr @ np.linalg.inv(U)

        # Solve the linear system to adjust the left eigenvectors
        M_diagonal = self.M.diagonal()
        matrix_A = np.zeros((self.M.shape[0], self.M.shape[0]), dtype=np.complex_)
        for i in range(self.M.shape[0]):
            matrix_A[i, :] = lam * vr_primary[i, :] * vl_primary[i, :]
        coefficients = solve(matrix_A, M_diagonal)
        vl = vl_primary * coefficients
        vr = vr_primary.copy()
        # Prepare secondary eigenvalues for output
        eigenvalues_Re_Im = np.stack((lam.real, lam.imag), -1)

        # Calculate the norm of the difference between M and its reconstruction to verify accuracy
        M_verifica3 = np.zeros(self.M.shape)
        for ij in range(self.M.shape[0]):
            M_verifica3 =M_verifica3 + lam[ij] * ( vr[:,ij][:,np.newaxis] @ (vl[:,ij][:,np.newaxis]).T )

        # Calculate the norm of the difference between M and its reconstruction to verify accuracy
        verification_error = np.linalg.norm(self.M - M_verifica3)
        print("Matrix Reconstruction Error:", verification_error)

        return lam, vl, vr, eigenvalues_Re_Im
     def get_transition_matrix_and_eigen(self):
        """
        Calculates the transition matrix for state changes over time based on the initialized
        microstate assignments and performs eigen decomposition to analyze the properties
        of the transition matrix, including calculating relaxation times.
        """
        self.M = self.transition_matrix()
        self.eigenvalues, self.left_eigenvectors, self.right_eigenvectors, self.eigenvalues_Re_Im = self.eigenvalue_decomposition()
        self.relaxation_times = -1 / np.log(np.abs(self.eigenvalues[1:]))
    
     def get_extreme_microstates(self, which_eigen):
        """
        Identify the most extreme microstates based on the specified eigenvector components.
    
        Parameters:
        - which_eigen (int): Index of the eigenvector to use for determining extremity.
    
        The method calculates the pairwise distances between the real and imaginary parts of the
        specified eigenvector's components and finds indices of the points that are furthest apart.
        """
        from scipy.spatial import distance
        # Stack the real and imaginary parts of the specified eigenvector components
        vl_2 = np.stack([self.left_eigenvectors[:, which_eigen].real, self.left_eigenvectors[:, which_eigen].imag], axis=1)
    
        # Calculate pairwise distances between these components
        max_dist = distance.cdist(vl_2, vl_2)
    
        # Find indices of the 3 most distant values
        threshold = np.sort(max_dist.ravel())[-6]  # Sort and pick threshold for top 3 distances
        inj = np.where(max_dist >= threshold)[0]
    
        # Sort these indices based on the real component of the eigenvector
        self.inj = inj[np.argsort(vl_2[inj, 0])]
    
        # Identify the closest data points to these extreme centroids
        self.idx = []
        for j in self.inj:
            distances = np.linalg.norm(self.data_tbc * self.scale_sqrt - self.centroids[j], axis=1)
            self.idx.append(np.argmin(distances))
            
     def visualize_microstate_mean_AIR2M(self, which_micro, color_local):
        """
        Plot atmospheric data using Basemap with specific overlays indicating different regions.
        - ia: Index to identify specific scenarios or microstates of interest to plot.
        - color_local: Color used for highlighting specific features.
        """
        fig = plt.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
        m = Basemap(projection='ortho', lat_0=90, lon_0=0, resolution='l')
        m.shadedrelief()
        m.drawlsmask(land_color='silver',ocean_color='white',lakes=True)
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(0,90+1,15.)  , labels=[1,0,0,0], fontsize=0, linewidth=1.5)
        m.drawmeridians(np.arange( 0. , 360 ,30.), labels=[0,0,0,1], fontsize=0, linewidth=1.5)

        # Data selection based on the microstate of interest 'ia'
        selected_indices = np.where(self.init_partition == which_micro)[0]
        tbp = self.AIR2M_filtered[selected_indices, :, :].mean(axis=0)

         # Create grid for data
        y = np.arange(0, 91, self.scale)
        x = np.arange(0,361, self.scale)
        longi = (list(range(0,144)) + list(range(1)))
        X, Y = np.meshgrid(x, y)
        if np.abs(tbp.mean())<100:
            p1 = -500
            p97 = 500
        else:
            p1 = 4550
            p97 = 5990
         
        dp = int((p97 - p1)/1000)
        
         
        levels1 = np.arange( p1, p97, dp )
        x1, y1 = m(X, Y)
        CS = plt.contourf(x1, y1, tbp[:,longi] ,levels = levels1, alpha=0.5, cmap='seismic' )

        if (self.longi[int(np.round( len( self.longi )/2 )) ]==0) & (len(self.longi)==49):
        
            x2, y2 = m(X[13:30,[-25,24]], Y[13:30,[-25,24]])
            plt.plot(x2[:,0], y2[:,0], c='limegreen', linewidth=3.5)
            plt.plot(x2[:,1], y2[:,1], c='limegreen', linewidth=3.5)
             
            x2, y2 = m(X[[13,29],:][:,range(-25,25,1)], Y[[13,29],:][:,range(-25,25,1)])
            plt.plot(x2[0,:], y2[0,:], c='limegreen', linewidth=3.5)
            plt.plot(x2[1,:], y2[1,:], c='limegreen', linewidth=3.5)
        elif (self.longi[int(np.round( len( self.longi )/2 )) ]==72) & (len(self.longi)==49):
            x2, y2 = m(X[13:30,[48,96]], Y[13:30,[48,96]])
            plt.plot(x2[:,0], y2[:,0], c='limegreen', linewidth=3.5)
            plt.plot(x2[:,1], y2[:,1], c='limegreen', linewidth=3.5)
             
            x2, y2 = m(X[[13,29],:][:,range(48,97,1)], Y[[13,29],:][:,range(48,97,1)])
            plt.plot(x2[0,:], y2[0,:], c='limegreen', linewidth=3.5)
            plt.plot(x2[1,:], y2[1,:], c='limegreen', linewidth=3.5)
        elif (self.longi[int(np.round( len( self.longi )/2 )) ]==0) & (len(self.longi)==25):
            x2, y2 = m(X[13:30,[-13,12]], Y[13:30,[-13,12]])
            plt.plot(x2[:,0], y2[:,0], c='limegreen', linewidth=3.5)
            plt.plot(x2[:,1], y2[:,1], c='limegreen', linewidth=3.5)
             
            x2, y2 = m(X[[13,29],:][:,range(-13,13,1)], Y[[13,29],:][:,range(-13,13,1)])
            plt.plot(x2[0,:], y2[0,:], c='limegreen', linewidth=3.5)
            plt.plot(x2[1,:], y2[1,:], c='limegreen', linewidth=3.5)
        elif (self.longi[int(np.round( len( self.longi )/2 )) ]==72) & (len(self.longi)==25):
            x2, y2 = m(X[13:30,[60,84]], Y[13:30,[60,84]])
            plt.plot(x2[:,0], y2[:,0], c='limegreen', linewidth=3.5)
            plt.plot(x2[:,1], y2[:,1], c='limegreen', linewidth=3.5)
               
            x2, y2 = m(X[[13,29],:][:,range(60,85,1)], Y[[13,29],:][:,range(60,85,1)])
            plt.plot(x2[0,:], y2[0,:], c='limegreen', linewidth=3.5)
            plt.plot(x2[1,:], y2[1,:], c='limegreen', linewidth=3.5)
      
        x2, y2 = m(X[[0,3],:][:,range(0,144,1)], Y[[0,3],:][:,range(0,144,1)])  
        plt.plot(x2[1,:], y2[1,:], c=color_local, linewidth=5)          
               
    
        plt.title('Geopotential Height at 500 hPa')
        plt.show()
     
