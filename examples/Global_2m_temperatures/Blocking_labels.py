#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:39:49 2022

@author: sspringe
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def Tibaldi_Molteni_indexes( psi, t ):
    # compute the Tibaldi Molteni index following eq5 of Luccarini Gritsun 2019
    
    GHGN = np.zeros( (psi.shape[ 2 ], 5) )
    GHGS = np.zeros( (psi.shape[ 2 ], 5) )
    
    # GHGN = np.zeros( (psi.shape[ 2 ], 3) )
    # GHGS = np.zeros( (psi.shape[ 2 ], 3) )
    
    
    latitudiniN = np.array(range(0,90,4))
    ran = np.array( range( 23 ) )
    indici = ran[ latitudiniN >= 0 ]

    
    for i in range( -2, 3, 1 ) :
        # for i in range( -1, 2, 1 ) :
        
        # Z_N = psi[ t,  20 + i, : ]
        # Z_0 = psi[ t,  15 + i, : ]
        # Z_S = psi[ t,  10 + i, : ]

        Z_N = psi[ t,  ran[ latitudiniN == 80 ] + i, : ]
        Z_0 = psi[ t,  ran[ latitudiniN == 60 ] + i, : ]
        Z_S = psi[ t,  ran[ latitudiniN == 40 ] + i, : ]
        
        # Z_N = psi[ t,  ran[ latitudiniN == 80 ] + i, : ]
        # Z_0 = psi[ t,  ran[ latitudiniN == 60 ] + i, : ]
        # Z_S = psi[ t,  ran[ latitudiniN == 40 ] + i, : ]
    
        # GHG geopotential height gradient
        GHGN[ :, i ] = ( Z_0 - Z_N ) / 0.006584
        
        GHGS[ :, i ] = ( Z_0 - Z_S )
        
        GHG  = ( GHGN > 1 ) & ( GHGS > 0 )
        
    
    return GHG.sum( axis = 1) > 0, GHGN.ravel(), GHGS.ravel()

def Tibaldi_Molteni_indexes2( psi, t ):
    # compute the Tibaldi Molteni index following eq5 of Luccarini Gritsun 2019
    
    GHGN = np.zeros( (psi.shape[ 2 ], 5) )
    GHGS = np.zeros( (psi.shape[ 2 ], 5) )
    
    latitudiniN = np.linspace( -90, 90, 73 )  
    ran = np.array( range( 73 ) )
    indici = ran[ latitudiniN >= 0 ]

    
    for i in range( -2, 3, 1 ) : #    55. ,  57.5,  60. ,  62.5,  65.
        
        Z_N = psi[ t,  ran[ latitudiniN == 80 ] - ran[ latitudiniN == 0 ] + i, : ]
        Z_0 = psi[ t,  ran[ latitudiniN == 60 ] - ran[ latitudiniN == 0 ] + i, : ]
        Z_S = psi[ t,  ran[ latitudiniN == 40 ] - ran[ latitudiniN == 0 ] + i, : ]
    
        # GHG geopotential height gradient
        GHGN[ :, i ] = ( Z_0 - Z_N ) 
        
        GHGS[ :, i ] = ( Z_0 - Z_S )
        
        GHG  = ( GHGN > 150 ) & ( GHGS > 0 )
        
    
    return GHG.sum( axis = 1) > 0, GHGN.ravel(), GHGS.ravel()


def SISSA_indexes1( psi_dot, t ):
    # compute the SISSA index1 using the 'S-N' 2 jump 'derivative'
    # consider adding a check about INDEX psi_dot.max == psi.max 
    psi_dot_t = psi_dot[ t, :, : ]
    
    MAX_V = psi_dot_t[ :26, : ].max( axis = 0 )
    MAX_I = np.argmax( psi_dot_t[ :26, : ]  , axis = 0)
    
    MIN_V = psi_dot_t.min( axis = 0 )
    MIN_I = np.argmin(psi_dot_t, axis = 0)
    
    MIN_V_left = np.zeros( (psi_dot.shape[ 2 ]) )
    MIN_I_left = np.zeros( (psi_dot.shape[ 2 ]) )
    
    # max derivative shold be high enough, test values
    C1 = MAX_V > 70
    # index of max should be located between 42.5 - 62.5 longitudine North
    C2 = (MAX_I*2.5 > 40) & (MAX_I*2.5 < 65)
    # The minimum should be deep enough
    C3 = MIN_V - MAX_V < -150
    # index of min should be located between 62.5 - 77.5 longitudine North
    C4 = (MIN_I*2.5 > 60) & (MIN_I*2.5 < 80)
    
    for i in range( psi_dot.shape[ 2 ] ):
        if MAX_I[i] <= 8:
            MIN_V_left[ i ] = psi_dot_t[ 0, i  ] # check if this makes damages
            MIN_I_left[ i ] = 0
        else:
            MIN_V_left[ i ] = psi_dot_t[ 8 : MAX_I[i]+1, i  ].min( )
            MIN_I_left[ i ] = np.argmin (psi_dot_t[ 8 : MAX_I[i]+1, i  ]) + 8
    
    C5 = MIN_V_left - MAX_V < -90
    
    Blocking = C1 & C2 & C3 & C4 & C5      
    
    return Blocking, MAX_I, MAX_V



def Check_TM_blocking( psi, time ):

    
    # to be used with clustering
    GHG_test = np.zeros( ( time, psi.shape[2] * 5*2 ) )
    # GHG_test = np.zeros( ( time, psi.shape[2] * 3*2 ) )
    Blocking = np.zeros( ( time, psi.shape[2] ) )

    for i in range( time ):
        Blocking[ i, : ],  GHG_test[ i, :5 * psi.shape[2] ], GHG_test[ i, 5 * psi.shape[2]: ] = Tibaldi_Molteni_indexes( psi, i )
        # Blocking[ i, : ],  GHG_test[ i, :3 * psi.shape[2] ], GHG_test[ i, 3 * psi.shape[2]: ] = Tibaldi_Molteni_indexes( psi, i )
        
    # Atlantic_Blocking = Blocking[ :, A_index ]
    # Pacific_Blocking  = Blocking[ :, P_index ]
        
    return Blocking, GHG_test

def Check_TM_blocking2( psi, time ):

    
    # to be used with clustering
    GHG_test = np.zeros( ( time, psi.shape[2] * 5*2 ) )
    
    Blocking = np.zeros( ( time, psi.shape[2] ) )
    
    for i in range( time ):
        Blocking[ i, : ],  A, B = Tibaldi_Molteni_indexes2( psi, i )
        
    return Blocking, GHG_test

def Check_TM_blockingSISSA( psi_dot, time ):

    
    # to be used with clustering
    MAX_I = np.zeros( ( time, psi_dot.shape[2] ) )
    MAX_V = np.zeros( ( time, psi_dot.shape[2] ) )    
    Blocking = np.zeros( ( time, psi_dot.shape[2] ) )
    
    for i in range( time ):
        Blocking[ i, : ], MAX_I[ i, : ], MAX_V[ i, : ] = SISSA_indexes1( psi_dot, i )
       
        
    return Blocking, MAX_I, MAX_V

def Get_relevant_indexes():
    
    # East before 180, West after 180
    longitudine = np.concatenate( ( np.array( range( 0, 181, 4 ) ), \
                                    np.array( range( 176, 0, -4 ) ) ) )
    # Atlantic blocking area 56W - 80E
    A_index = np.concatenate( (np.array( range ( 76, 90 ) ),  \
                                np.array( range ( 0, 21 ) ) ) )
    # Pacific blocking area 104E - 90W
    P_index = np.array( range ( 26, 67 ) )
    
    # index of interest used to detect the blocking events
    indRel = np.array( [ 9, 10, 11, 14, 15, 16, 19, 20, 21 ] ) 
    
    return A_index, P_index, indRel


#%%
from IPython.display import display
def Remove_lenth_1( Blocking ):
    # remove blocks of lenth 1
    time = Blocking.shape[0]
    auxx = (Blocking[ 0 , : ] == 1) & (Blocking[ 1 , : ] == 0)
    Blocking[ 0, auxx ] = 0
    
    auxx = (Blocking[ -1 , : ] == 1) & (Blocking[ -2 , : ] == 0)
    Blocking[ -1, auxx ] = 0
    
    for j in range( 1, time - 1 ):  
        
        auxx = (Blocking[ j - 1 , : ] == 0) & (Blocking[ j , : ] == 1) & \
            (Blocking[ j + 1 , : ] == 0)
        Blocking[ j, auxx ] = 0
    
    return Blocking


#%%
import itertools
import operator

def Check_TM_blocking_len( Blocking ):
    #
    Index_list = []
    Blocking_LLL = np.zeros(Blocking.shape)
    # group the block based on their index
    for ind in range( Blocking.shape[1] ):
        auxy = [[i for i,value in it] for key,it in \
                itertools.groupby(enumerate( Blocking[ :, ind ] ), \
                                  key=operator.itemgetter(1)) if key != 0]
        # define the length in the matrix form    
        for jnd in range( len( auxy ) ):
            Blocking_LLL[ auxy[ jnd ] , ind ] = len( auxy[ jnd ] )
            
        Index_list.append( auxy )
        
    
    return Index_list, Blocking_LLL
