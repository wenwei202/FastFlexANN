//
//  neuron_t.h
//  FastFlexANN
//
//  Created by Wesley on 3/10/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#ifndef FastFlexANN_neuron_t_h
#define FastFlexANN_neuron_t_h
#include "common.h"

/**
 * @struct information for a neuron
 */
struct neuron_t{
    //int ID;
    fflex_msg_t sum; //sum of weighted inputs
    fflex_msg_t activation; //activation of sum
    fflex_msg_t bp_msg; //derivative for bp
    
    int *back_neuron_idx; //indices of connected neurons in previous layer
    int *back_weight_idx; //weights on connections with previous layer
    int back_connection_num;//number of connections with previous layer
    
    int *forward_neuron_idx;//indices of connected neurons in next layer
    int *forward_weight_idx;//weights on connections with next layer
    int forward_connection_num;//number of connections with next layer
};

#endif
