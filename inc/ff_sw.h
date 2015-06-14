//
//  ff_sw.h
//  FastFlexANN
//
//  Created by Wesley on 3/12/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#ifndef FastFlexANN_ff_sw_h
#define FastFlexANN_ff_sw_h

struct ff_sw_t{
    int *cluster_size_lib;
    int size_num;
    float p;
    int (*pair)[2];
    int pair_num;
    int neuron_num1;
    int neuron_num2;
    
};

struct ff_sw_t * ff_sw_malloc();
void ff_sw_free(struct ff_sw_t * sw);
void ff_sw_get(struct ff_sw_t * sw);
#endif
