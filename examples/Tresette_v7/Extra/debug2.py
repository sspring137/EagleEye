#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:32:49 2025

@author: sspringe
"""




def test_assignment(test_data, reference_data, subset_indices, subset_indices_rev, K_M, NUMBER_CORES, PARTITION_SIZE ):
    

        import From_data_to_binary_post
        binary_sequences_ta, neighborhood_idx_ta = From_data_to_binary_post.create_binary_array_cdist_post_subset(
            test_data,
            reference_data,
            subset_indices,
            num_neighbors=K_M,
            num_cores=NUMBER_CORES,
            validation=None,
            partition_size=PARTITION_SIZE
        )


        binary_sequences_ta_rev, neighborhood_idx_ta_rev = From_data_to_binary_post.create_binary_array_cdist_post_subset(
            reference_data,
            test_data,
            subset_indices_rev,
            num_neighbors=K_M,
            num_cores=NUMBER_CORES,
            validation=None,
            partition_size=PARTITION_SIZE
        )



qt=0


overdensity_indicies_plus = get_indicies(np.quantile(result_dictionary['Upsilon_i_plus_null'],  CRITICAL_QUANTILES[qt]),result_dictionary['unique_elements_overdensities'])
overdensity_indicies_plus.sort()


overdensity_indicies_minus = get_indicies(np.quantile(result_dictionary['Upsilon_i_plus_null'],  CRITICAL_QUANTILES[qt]),result_dictionary['unique_elements_underdensities'])
overdensity_indicies_minus.sort()

i_o_d = np.where(result_dictionary['stats']['Upsilon_i_plus']>result_dictionary['Upsilon_star_plus'][0])[0]
i_o_dr = np.where(result_dictionary['stats']['Upsilon_i_Val_plus']>result_dictionary['Upsilon_star_plus'][0])[0]

import From_data_to_binary_post
binary_sequences_ta, neighborhood_idx_ta = From_data_to_binary_post.create_binary_array_cdist_post_subset(
    test_data,
    reference_data,
    i_o_d,
    num_neighbors=20,
    num_cores=NUMBER_CORES,
    validation=None,
    partition_size=PARTITION_SIZE
)


binary_sequences_ta_rev, neighborhood_idx_ta_rev = From_data_to_binary_post.create_binary_array_cdist_post_subset(
    reference_data,
    test_data,
    i_o_dr,
    num_neighbors=20,
    num_cores=NUMBER_CORES,
    validation=None,
    partition_size=PARTITION_SIZE
)






neighborhood_idx_ta[binary_sequences_ta==0]         = neighborhood_idx_ta[binary_sequences_ta==0]  + test_data.shape[0]

neighborhood_idx_ta_rev[binary_sequences_ta_rev==1] = neighborhood_idx_ta_rev[binary_sequences_ta_rev==1]  + test_data.shape[0]



cc = []
for jj in range( neighborhood_idx_ta.shape[0] ):
    i_i_i = list(set(overdensity_indicies_plus).intersection(set(neighborhood_idx_ta[jj,1:])))
    i_i_t  = neighborhood_idx_ta[jj,0]
    

    aa = result_dictionary['stats']['Upsilon_i_plus'][i_i_t]
    
    bb = result_dictionary['stats']['Upsilon_i_plus'][i_i_i]
    
    cc.append( (aa>=bb).any() )




mask_1 = 

























