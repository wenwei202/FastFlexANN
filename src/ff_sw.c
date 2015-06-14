//
//  ff_sw.c
//  FastFlexANN
//
//  Created by Wesley on 3/12/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#include <stdio.h>
#include "../inc/ff_sw.h"
#include "common.h"

struct ff_sw_t * ff_sw_malloc(){
    struct ff_sw_t * res = (struct ff_sw_t *)malloc(sizeof(struct ff_sw_t));
    res->p = 0;
    int min = 16;
    int max = 64;
    int step = 4;
    res->size_num = (max-min+step)/step;
    res->cluster_size_lib = (int *)malloc(sizeof(int)*res->size_num);
    for (int i=0; i<res->size_num; i++) {
        res->cluster_size_lib[i] = min + i*step;
    }
    
    res->pair = NULL;
    res->pair_num = 0;
    
    res->neuron_num1 = 0;
    res->neuron_num2 = 0;
    return res;
}

void ff_sw_free(struct ff_sw_t * sw){
    if (!sw) {
        return;
    }
    if (sw->size_num>0 && sw->cluster_size_lib) {
        free(sw->cluster_size_lib);
    }
    if (sw->pair && sw->pair_num>0) {
        for (int i=0; i<sw->pair_num; i++) {
            free(*(sw->pair+i));
        };
        free(sw->pair);
    }
    sw->pair_num = 0;
}


void ff_sw_get(struct ff_sw_t * sw){
    
}