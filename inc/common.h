//
//  common.h
//  FastFlexANN
//
//  Created by Wesley on 3/10/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#ifndef FastFlexANN_common_h
#define FastFlexANN_common_h
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

typedef double fflex_msg_t;
typedef int fflex_class_t;
typedef unsigned char uint8;

//tan activation function
#define TANSIGMOID(x) (1.7159*tanh(0.66666667*(x)))
//derivative 
#define DTANSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))
#undef DEBUG
//#define DEBUG

//activation for the output layer
enum output_activation_func{
    SOFT_MAX=0,
    TAN_SIGMOID
};

//the method to prune connections
enum weight_prune_method{
    STATIC=0,
    DYNAMIC // to be implemented in future
};

//random permutation
void randperm(int *input, int len);

//find elements with k smallest absolute values
fflex_msg_t find_kth_abs_smallest_elem(fflex_msg_t *input, int len,int k);

void find_max(fflex_msg_t *input, int len, fflex_msg_t *maxvalue, int *idx);
int find_max_int(int *input, int len);
int is_file_exist(char *file);
#endif
