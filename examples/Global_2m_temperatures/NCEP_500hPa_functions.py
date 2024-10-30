#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:04:17 2022

@author: sebas
"""
import numpy as np
# import numdifftools as nd
import matplotlib.pyplot as plt
# 
#####################
# from numbalsoda import lsoda_sig, lsoda # numbalsoda
# import numba as nb
# import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.insert(2, '/scratch/sspringe/MSM/Tools_MSM')
# sys.path.insert(1, '/home/sebas/Documents/GitHub/MSM/Tools_MSM')

# import matplotlib
# import Blocking_labels 
# import Denoising_procedure 
# import Clustering_procedure 
# import Plotting_procedure 
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.pyplot import cm
# # from sklearn.utils import resample
# # import faiss
# # import pandas as pd
# # from fastdist import fastdist

from sklearn.cluster import kmeans_plusplus

from scipy.linalg import eig
from scipy.linalg import lu
from scipy.linalg import solve
# apparently this avoid an error due to incompatibility between scipy and torch
_, _,_ =eig( np.eye(2), left = True )
import torch
#%% solvers


     
    
def dist_weights( longi, lati, scale, grid_discretization,filters ):
      from scipy import signal
     
      if any(ele == 'Lati_area' for ele in filters):
          weights1 = np.cos( np.deg2rad( ( np.array( lati ) * scale ) ) ) 
      else:
          weights1 = np.ones(len(lati))
    
      if any(ele == 'Longi_Gaussian' for ele in filters):
        window = signal.gaussian(len(longi), std=len(longi)/2)#np.sqrt(len(longi)*10))   #std=np.sqrt(len(longi)/2))
      else:
        window = np.ones(len(longi))
         
      grid_filter = np.zeros( ( grid_discretization[0], len(longi) ) )
     
      final_weights = []
      for jj in range(window.shape[0]):
          final_weights.extend( weights1*window[jj] )
          grid_filter[lati,jj] = ( weights1*window[jj] )
     
      final_weights = np.sqrt(np.array( final_weights ))
     

      return final_weights,grid_filter    




# def get_initial_centroids(indata_Km, n_clusters, scale_sqrt, *args ):
    
#     init_centroids = indata_Km[ np.random.permutation( range( indata_Km.shape[ 0 ] ) )[ :n_clusters ], : ]
#     init_index = 1
#     if args:
#         # in case of given input we keep the given part 
#         init_centroids_temp = args[0]
#         init_index = init_centroids_temp.shape[0]
#         init_centroids[ :init_index, : ] = init_centroids_temp
#         # fastdist.sqeuclidean, "sqeuclidean"
#     for ij in range(init_index,n_clusters):
#         if ij==init_index:
#             a1t = torch.tensor( indata_Km*scale_sqrt )
#             b1t = torch.tensor( init_centroids[:ij,:]*scale_sqrt )
#             dist =torch.cdist( a1t, b1t , p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
#         else:
#             c1t = torch.tensor(init_centroids[ij,:][np.newaxis,:]*scale_sqrt)
#             dist1 =torch.cdist( a1t, c1t , p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
#             dist = np.concatenate( (dist, dist1 ), axis=1 )
#         init_centroids[ij,:]=indata_Km[ np.random.choice(range(len(indata_Km)), size=1, p=(dist.sum(axis=1)**3)/(dist.sum(axis=1)**3).sum()), :]
#         while np.unique(init_centroids[:ij+1,:], axis=0).shape[0]<ij+1:
#             init_centroids[ij,:]=indata_Km[ np.random.choice(range(len(indata_Km)), size=1, p=(dist.sum(axis=1)**4)/(dist.sum(axis=1)**4).sum()), :]
#             # display('ancora')
              
#         if ij%10==0:
#             display(ij)
    
#     return init_centroids

def WEIGHTED_KMEANS(indata_Km, n_clusters,Dm, scale_sqrt, *args):
    
    
    # if args==True:            
    #     init_centroids = get_initial_centroids(indata_Km, n_clusters-1, scale_sqrt)
    #     init_centroids = np.concatenate( (init_centroids, np.zeros((1,Dm))) , axis=0)
    #     display(init_centroids)
    # else:
    # init_centroids = get_initial_centroids(indata_Km, n_clusters, scale_sqrt)
    init_centroids, indices = kmeans_plusplus(indata_Km*scale_sqrt, n_clusters=n_clusters, n_local_trials=100000)

    centroids = init_centroids.copy()
    maxiter = 700
    
    indata_Km_torch = torch.tensor(indata_Km*scale_sqrt)
    
    for itr in range(maxiter):
        
        centroids_torch = torch.tensor(centroids)
        distances_to_centroids =  torch.cdist( indata_Km_torch, centroids_torch , p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        
        cluster_assignment = np.argmin(distances_to_centroids, axis = 1)
    
        new_centroids = np.array([indata_Km[cluster_assignment == i].mean(axis = 0) for i in range(n_clusters)]) *scale_sqrt       
        # if the updated centroid is still the same,
        # then the algorithm converged
        if np.all(centroids == new_centroids):
            break
        # display(itr)
        
        centroids = new_centroids

    init_cluster = np.array(cluster_assignment)
    return init_cluster, centroids


        
def Initial_partition( indata_K, longi, Blocking_LLL, n_clusters ) :
    
    
    # n_clusters = 2000    # initial number of clusters
    
    tau = 1             # time lag
    
    # cluster the data by the Kmeans method
    lati = range(13,29+1)
    scale = 360/Blocking_LLL.shape[1]
    grid_discretization = [37,144]
    filters = [ "Lati_area", "Longi_Gaussian" ]
    scale_sqrt, grid_filter = dist_weights( longi, lati, scale, grid_discretization,filters )

    indata_K_tbc = indata_K[:,lati,:][:,:,longi]
    
    indata_Km = indata_K_tbc.reshape((indata_K_tbc.shape[0],np.prod(indata_K_tbc.shape[1:])), order='F').copy(order='C')
    init_clust,centroids =   WEIGHTED_KMEANS( indata_Km , n_clusters, np.prod(indata_K_tbc.shape[1:]), scale_sqrt )
    
    # number of elements per cluster
    aaa = np.zeros( n_clusters )

    for i in range( n_clusters):
        aaa[i] = ( init_clust ==i ).sum()
              
    return init_clust, aaa, n_clusters, centroids, lati, grid_filter, scale_sqrt       
        
        
def transition_matrix( data, n_clusters, tau, doPlot ):
    
    t = np.array( data )
    tau = tau
    total_inds = t.size - ( tau + 1) + 1
    t_strided = np.lib.stride_tricks.as_strided(
                                    t,
                                    shape = ( total_inds, 2 ),
                                    strides = ( t.strides[ 0 ], tau * t.strides[ 0 ] ) )
    
    #%% remove 28th of FEB -> 1st of DIC &similar for tau = 2
    my_list = list(range(0,t_strided.shape[0]))
    if tau==1:
        indexes = list(range(89,t_strided.shape[0],90)) 
        for index in indexes:
            my_list.remove(index)
    if tau==2:
        indexes = np.sort(list(range(88,t_strided.shape[0],90)) + list(range(89,t_strided.shape[0],90)))
        for index in indexes:
            my_list.remove(index)
    t_strided = t_strided[my_list,:]
    #%% continue after 28th of FEB -> 1st of DIC have been removed
    inds, counts = np.unique( t_strided, axis = 0, return_counts = True )

    P = np.zeros( ( n_clusters, n_clusters ) )
    P[ inds[:, 0], inds[:, 1] ] = counts
    
    sums = P.sum(axis = 1)
    # Avoid divide by zero error by normalizing only non-zero rows
    P[ sums != 0 ] = P[ sums != 0 ] / sums[ sums != 0 ][ :, None ]
    
    # P = P / P.sum(axis = 1)[:, None]
    
    # plot the transition matrix
    if doPlot:
        plt.figure()
        plt.imshow( P, aspect = 'auto', cmap = 'jet' ) #, vmin = 0, vmax = 1 )
        plt.title( ' Transition probability matrix ' )
        plt.colorbar()
        plt.show()
    
    
    return P


def eigen_decomposition( M ):
    

    lam, vl, vr =eig( M, left = True )
    idx = np.abs( lam ).argsort()[ ::-1 ]   
    lam = lam[ idx ]
    vl = vl[ :, idx ].conj() # in colonna
    vr = vr[ :, idx ] # in colonna

    # normalization
    vr[:,0] = vr[:,0] / vr[0,0]  # normalize the vector
    c = sum( vr[:,0] * vl[:,0]  ).real
    vl[:,0] = vl[:,0] / c # normalize the vector

    test3 =np.zeros((M.shape[0],M.shape[0]), dtype=np.complex_)
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            test3[i,j]=sum( vr[:,i] * vl[:,j].T  )

    P, L, U = lu(test3)


    vl_prim = np.linalg.inv(L) @ vl
    vr_prim = vr @ np.linalg.inv(U) 



    M_t = M.diagonal()
    AA = np.zeros((M.shape[0],M.shape[0]),dtype=np.complex_)
    for ij in range(M.shape[0]):
        AA[ij,:] = (lam * vr_prim[ij,:]*vl_prim[ij,:] )
        

    bb = solve(AA,M_t)


    vl = vl_prim * bb

    vr = vr_prim.copy()
    
    
    # group the eigenvalues using clustering
    # w_2 = np.stack( ( w.real, np.abs( w.imag ) ), -1 )
    w_2 = np.stack( ( lam.real,  lam.imag  ), -1 )
    w = lam.copy()
    
    
    T_verifica3 = np.zeros(M.shape)
    for ij in range(M.shape[0]):
        T_verifica3 =T_verifica3 + lam[ij] * ( vr[:,ij][:,np.newaxis] @ (vl[:,ij][:,np.newaxis]).T )
        # T_verifica =T_verifica + lam[ij] * np.dot( vr[:,ij][:,np.newaxis] , (vl[:,ij][:,np.newaxis]).T )
    display("T - T_verifica")    
    display(np.linalg.norm( M - T_verifica3))
    
    return w, vl, vr, w_2


def plot_relaxation_times_eigenvalues( w, w_2, tau, n_clusters ):
    
    # plot the relaxation times
    plt.figure()
    plt.plot( np.array( range( 1, n_clusters ) ), -tau / ( np.log( np.abs( w[ 1: ] ) ) ), marker='.', linestyle = 'None' )
    plt.title( '  relaxation times for tau=1'  )
    plt.show()
    
    plt.figure()
    plt.plot( w_2[ :, 0 ], w_2[ :, 1 ] , marker='.', linestyle = 'None' )
    plt.ylim([-0.05,1.05])
    plt.xlim([-0.05,1.05])
    plt.grid()
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title( '  Eigenvalues in 2D '  )
    plt.show()

def plot_relaxation_times_eigenvalues_final_comparison( w, w_final, tau ):
    
    # plt.figure()
    # plt.plot( np.array( range( 1, 11 ) ), -tau / ( np.log( np.abs( w[ 1:11 ] ) ) ), marker='.', linestyle = 'None' )
    # plt.plot( np.array( range( 1, w_final.shape[0] ) ), -tau / ( np.log( np.abs( w_final[ 1: ] ) ) ), marker='.', linestyle = 'None' )
    # plt.title( '  relaxation times for tau=1'  )
    # plt.show()
    w_2 = np.stack( ( w.real,  w.imag ), -1 )
    w_2f = np.stack( ( w_final.real,  w_final.imag ), -1 )
    
    BB = -tau / ( np.log( np.abs( w[ 1: ] ) ) )
    Omega= np.zeros(w.shape)
    
    i0 = np.append( False,w_2[1:,0]>0 )
    i1 = np.append( False,(w_2[1:,0]<0) & (w_2[1:,1]>=0) )
    i2 = np.append( False,(w_2[1:,0]<0) & (w_2[1:,1]<0) )
    
    Omega[i0] = 2*np.pi/np.abs(np.arctan(w_2[i0,1]/w_2[i0,0]))
    Omega[i1] = 2*np.pi/np.abs(np.arctan(w_2[i1,1]/w_2[i1,0])+np.pi)
    Omega[i2] = 2*np.pi/np.abs(np.arctan(w_2[i2,1]/w_2[i2,0])-np.pi)
    
    Omega[Omega>1e6] =-1
    
    OmegaF = 2*np.pi/np.arctan(w_2f[1:,1]/w_2f[1:,0])
    OmegaF[OmegaF>1e6] =-1
    
    plt.figure()
    plt.plot( BB[:20], Omega[1:21],marker='o', linestyle = 'None')
    # plt.plot( BB, Omega[1:], marker='.', linestyle = 'None' )
    plt.plot( -tau / ( np.log( np.abs( w_final[ 1: ] ) ) ),OmegaF , marker='s', linestyle = 'None' )
    plt.grid()
    plt.xlabel('relaxation times', fontsize=15)
    plt.ylabel('T', fontsize=15)   
    # plt.title( '  relaxation times for tau=1', fontsize=15  )
    plt.legend(['MICRO','MACRO'], fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylim([-11.05,Omega[1:21].max()*1.13])
    plt.xlim([-1.05,BB[0]*1.13])
    plt.show()

def get_relevant_portion_eigenv( w_2, vl, relevan_portion_eigenvector ):
    
    unique_rows, indeces= np.unique( w_2, axis = 0, return_index = True )

    # remove the first and skip the conj pairs
    indeces1 = np.sort(indeces)[ 1 : relevan_portion_eigenvector + 1 ]
    vl_2 = np.concatenate( ( vl[ :, indeces1 ].real,  vl[ :, indeces1 ].imag  ),1)
    
    vl_3 = (vl_2.T/np.abs(vl_2).max(axis=1)).T
    
    return vl_2, vl_3


def plot_eigenv_clustered(vl_n, vl_n_labelled, n_clusters2 ):
    
    # define the colours used in the next plot
    color = iter( cm.tab20( np.linspace(0, 1, 20 ) ) ) 

    # plot the eigenvectors based on the cluster assigned
    plt.figure()
    for jj in range( n_clusters2  ):
        c = next( color )
        plt.plot( vl_n[  vl_n_labelled == jj, : ].T, c = c )
        plt.grid( True )
        plt.title('Clusters with relevent eigenvector parts ')
    plt.show()


def assign_small_clusters_to_big_ones(indata_K, init_clust, vl_n_labelled, n_clusters, n_clusters3, longi, Blocking_LLL):

    kml = init_clust.copy()
    #%% create a list containing the indeces of the clusters grouped together
    clauster_list = []
    
    ininin = np.array( range( n_clusters) ) ## just as a support
    
    for jjj in range( n_clusters3 ):
        
        clauster_list.append( ininin[ ( vl_n_labelled== jjj ) ] )
    
        for kkk in range( clauster_list[ jjj ].shape[0] ):
            
            kml[ init_clust ==  clauster_list[ jjj ][ kkk ] ] = -jjj
    
    # array containing the respective label of the "BIG" clusters for each data point
    kml = np.abs( kml )
    
    #%% plot the data based on the "BIG" clusters
    
    return kml, clauster_list

def check_halo(vl_n):
    
    aaaa = np.sort( (np.abs(vl_n)).mean(axis = 1))
    bbbb = ( (aaaa < aaaa.max()*.1).sum()  )/aaaa.shape[0]
    
    if bbbb < .05:
        halo = 0
    else:
        halo=1
        
    return halo


def core_state_analisis( vl_n, vl_n_labelled, kml, n_clusters2, n_clusters3, doPlot  ):
    
    if n_clusters3 > n_clusters2:
        ##% to set all the elements not properly clustered to the proper cluster ( core state analysis )
        idx_vl_2_mean = np.zeros( n_clusters2 )
        for ij in range( n_clusters2  ):
            idx_vl_2_mean[ij] = np.abs(vl_n[ vl_n_labelled==ij ]).mean()
            
        idx_vl_2_min = np.where(idx_vl_2_mean == idx_vl_2_mean.min())[0][0]
        kml_core_state = kml + 0
    
        if kml_core_state[ 0 ] == idx_vl_2_min:
            kml_core_state[ 0 ]  = kml_core_state[ ~(kml_core_state==idx_vl_2_min) ][0]
    
        #assign all the  idx_vl_2_min cluster elements to the other clusters
        for ij in list(np.where(kml_core_state == idx_vl_2_min)[0]):
            kml_core_state[ ij ] = kml_core_state[ ij -1 ] 
        
        
        if  doPlot:
            plt.figure()
            plt.plot( kml[:100] )
            plt.plot( kml_core_state[:100] )    
            plt.show()   
        
        if idx_vl_2_min > 0:
            kml_core_state[ kml_core_state > idx_vl_2_min ] += -1
            idx_vl_2_min = n_clusters3
        else:
            kml_core_state +=-1
            
    else:
        kml_core_state = kml
    
    list_clusters = list(range( n_clusters2 ))
    
    
    PT_list = []

    ## create a list of lists containing the informations
    for jj in range( kml_core_state.max()+1 ):
        PT_list.append([])

    PT_list[ kml_core_state[0] ].append( 0 )

    # add the first passage times 
    for j in range(1, kml_core_state.shape[0]):
        if ( kml_core_state[j-1] - kml_core_state[j] ) != 0 : 
            PT_list[ kml_core_state[j] ].append( j )

    FPT = []   
    MFPT = []     
    for jj in range( kml_core_state.max()+1 ):        
        FPT.append( np.array(PT_list[ jj ][1:])-np.array(PT_list[ jj ][:-1])  )
        MFPT.append( np.mean( FPT[jj] ) )
        

    display( 'Mean first passage time:' )
    display(MFPT)
    
    return kml_core_state, list_clusters, FPT, MFPT



def Kmeans_fixed0(vl_2, n_clusters3, halo_perc):

    labelsss = np.zeros( vl_2.shape[0] )
    bbb = np.linalg.norm( vl_2, axis = 1 )
    ccc = np.where( bbb >  np.percentile(bbb, halo_perc) ) #halo_perc = 45
    vl2_M = vl_2[ccc[0],:]
    kmeans = KMeans(n_clusters=n_clusters3-1, init='k-means++', n_init=10000, max_iter=500, ).fit(vl2_M)
    
    labelsss[ccc[0]] = kmeans.labels_+1
    labelsss = labelsss.astype(int)
    centroids = np.concatenate( (np.zeros( (1,vl_2.shape[1]) ) , kmeans.cluster_centers_) , axis = 0 ) 
    
    # kmeans2 = KMeans(n_clusters=n_clusters3, init=centroids, n_init=1, max_iter=300, ).fit(vl_2)
    
    
    return labelsss, centroids



def separate_areas( Blocking_LLL, idx, longi):           
    
    PT_list = []
    PT_list.append([])
    ddata = np.where((Blocking_LLL[idx, longi]>0))[0]
    # ddata = np.append(ddata, ddata+50)
    PT_list[0].append(ddata[0])
    count = 0
    for j in range(1,len(ddata)):
        if ( ddata[j] - ddata[j-1] ) == 1 : 
            PT_list[ count ].append( ddata[j] )
        else:
            PT_list.append([]) 
            count = count + 1
            PT_list[ count ].append( ddata[j] )
    
    return PT_list

def plot_contour(indata_K,psi_w,init_cluster,longi,lati, inj, centroids, Blocking_LLL, grid_filter, scale_sqrt):
    tbp = psi_w
    import matplotlib.cm as cm
    if longi[0] > longi[-1]:
        x = np.arange((longi[0]-144)*2.5, longi[-1]*2.5+1, 2.5)
    else:
        x = np.arange(longi[0]*2.5, longi[-1]*2.5+1, 2.5)
    
    y = np.arange(0, 91, 2.5)
    X, Y = np.meshgrid(x, y)
    
    indata_K_tbc = indata_K[:,lati,:][:,:,longi]
    
    indata_Km = indata_K_tbc.reshape((indata_K_tbc.shape[0],np.prod(indata_K_tbc.shape[1:])), order='F').copy(order='C')
    p0 = np.percentile(tbp, 17.4)
    p1 = np.percentile(tbp, 99.55)
    dp = int((np.percentile(tbp, 99.55) - np.percentile(tbp, 17.4))/20)
    
    for ggg in range(1,len(inj)):
        
        LL2 = np.linalg.norm(indata_Km*scale_sqrt-centroids[inj[ggg]], axis = 1)
        idx =  np.where(LL2 == LL2.min())[0][0]
        
        checkB =(Blocking_LLL[idx, longi]>0).sum()
        
        plt.figure()
        # plt.axhspan(29*2.5, 90, facecolor='0.5', alpha=0.5)
        # plt.axhspan(0, 13*2.5, facecolor='0.5', alpha=0.5)
        plt.contourf(X, Y, grid_filter, cmap=cm.Greys_r, vmin = 0, vmax = 1, levels = np.arange(0,1,0.05 ), alpha=0.4)
        
        if checkB>0:
            areas = separate_areas( Blocking_LLL, idx, longi)
            for dfdf in range(len(areas)):
                plt.axvspan(x[ areas[dfdf][0] ] , x[ areas[dfdf][-1] ], color='red', alpha=0.1)
               
        CS = plt.contour(X, Y, tbp[idx,:,:][:,longi] ,levels = np.arange( p0, p1, dp ), linewidths=1.5 )
        plt.clabel(CS, inline=2, fontsize=10)
        plt.title('Cluster centers ' + str(inj[ggg])+ '_' + str(idx) )
        plt.xlabel('LONGITUDINE')
        plt.ylabel('LATITUDINE')
        plt.show()
        

        


def plot_contour_2(indata_K,psi_w,init_cluster,longi,lati, inj, Blocking_LLL, grid_filter):
    
    tbp = psi_w
    import matplotlib.cm as cm    
    if longi[0] > longi[-1]:
        x = np.arange((longi[0]-144)*2.5, longi[-1]*2.5+1, 2.5)
    else:
        x = np.arange(longi[0]*2.5, longi[-1]*2.5+1, 2.5)
    
    y = np.arange(0, 91, 2.5)
    X, Y = np.meshgrid(x, y)
    p0 = np.percentile(tbp, 17.4)
    p1 = np.percentile(tbp, 99.55)
    dp = int((np.percentile(tbp, 99.55) - np.percentile(tbp, 17.4))/20)
    
    for ggg in range(1,len(inj)):
        
        idx =  np.where(init_cluster==inj[ggg])[0]
        
        for kkk in range(idx.shape[0]):
            checkB =(Blocking_LLL[idx[kkk], longi]>0).sum()
            plt.figure()
            # plt.axhspan(29*2.5, 90, facecolor='0.5', alpha=0.5)
            # plt.axhspan(0, 13*2.5, facecolor='0.5', alpha=0.5)
            plt.contourf(X, Y, grid_filter, cmap=cm.Greys_r, vmin = 0, vmax = 1, levels = np.arange(0,1,0.05 ), alpha=0.4)
            # plt.colorbar()
            
            if checkB>0:
                areas = separate_areas( Blocking_LLL, idx[kkk], longi)
                for dfdf in range(len(areas)):
                    plt.axvspan(x[ areas[dfdf][0] ] , x[ areas[dfdf][-1] ], color='red', alpha=0.1)
                   
            CS = plt.contour(X, Y, tbp[idx[kkk],:,:][:,longi] ,levels = np.arange( p0, p1, dp ), linewidths=1.5 )
            plt.clabel(CS, inline=2, fontsize=10)
            plt.title('Cluster centers ' + str(inj[ggg]) + '_' + str(idx[kkk]))
            plt.xlabel('LONGITUDINE')
            plt.ylabel('LATITUDINE')
            plt.show()



def plot_contour3(indata_K,psi_w,init_cluster,longi,lati, inj, centroids, Blocking_LLL, grid_filter, scale_sqrt, rows, cols, name):
    tbp = psi_w
    import matplotlib.cm as cm    
    if longi[0] > longi[-1]:
        x = np.arange((longi[0]-144)*2.5, longi[-1]*2.5+1, 2.5)
    else:
        x = np.arange(longi[0]*2.5, longi[-1]*2.5+1, 2.5)
    
    y = np.arange(0, 91, 2.5)
    X, Y = np.meshgrid(x, y)
    
    indata_K_tbc = indata_K[:,lati,:][:,:,longi]
    
    indata_Km = indata_K_tbc.reshape((indata_K_tbc.shape[0],np.prod(indata_K_tbc.shape[1:])), order='F').copy(order='C')
    from math import log10, floor
    p0 = np.percentile(tbp, 8.15)
    p1 = np.percentile(tbp, 99.55)
    
    p0 = round(p0, -int(floor(log10(abs(p0)))))
    p1 = round(p1, -int(floor(log10(abs(p1)))))
    
    dp = (p1 - p0)/20
    dp = round(dp, -int(floor(log10(abs(dp)))))
    
    fig1, axs1 = plt.subplots(rows,cols, figsize=(18, 9), facecolor='w', edgecolor='k')
    fig1.subplots_adjust(hspace = 1, wspace=.5)
    fig1.suptitle( '[' + str( longi[ np.floor(len(longi)/2).astype(int) ] *2.5) +']' )
    axs1 = axs1.ravel()
    local_i = 0
    for ggg in range(1,len(inj)):
        
        LL2 = np.linalg.norm(indata_Km*scale_sqrt-centroids[inj[ggg]], axis = 1)
        idx =  np.where(LL2 == LL2.min())[0][0]
        
        checkB =(Blocking_LLL[idx, longi]>0).sum()
        
        
        # axs1[local_i].contourf(X, Y, grid_filter, cmap=cm.Greys_r, vmin = 0, vmax = 1, levels = np.arange(0,1,0.05 ), alpha=0.4)
        # axs1[local_i].set_xlabel('Real', fontsize=15)
        # axs1[local_i].set_ylabel('Imag', fontsize=15)  
        # axs1[local_i].set_title( '[' + str(longi[6]*4) + ']', fontsize=15  )
        # axs1[local_i].tick_params(axis='both', which='major', labelsize=15)
        # axs1[local_i].grid()
       
        
        # plt.figure()
        # plt.axhspan(29*4., 90, facecolor='0.5', alpha=0.5)
        # plt.axhspan(0, 13*4., facecolor='0.5', alpha=0.5)
        axs1[local_i].contourf(X, Y, grid_filter, cmap=cm.Greys_r, vmin = 0, vmax = 1, levels = np.arange(0,1,0.05 ), alpha=0.4)
        
        if checkB>0:
            areas = separate_areas( Blocking_LLL, idx, longi)
            for dfdf in range(len(areas)):
                axs1[local_i].axvspan(x[ areas[dfdf][0] ] , x[ areas[dfdf][-1] ], color='red', alpha=0.1)
               
        CS = axs1[local_i].contour(X, Y, tbp[idx,:,:][:,longi] ,levels = np.arange( p0, p1, dp ), linewidths=1.5 )
        axs1[local_i].clabel(CS, inline=2, fontsize=10)
        axs1[local_i].set_title('Cluster centers ' + str(inj[ggg])+ '_' + str(idx) )
        axs1[local_i].set_xlabel('LONGITUDINE')
        axs1[local_i].set_ylabel('LATITUDINE')
        local_i = local_i+1
    plt.savefig(name +'.png')
    plt.close()




