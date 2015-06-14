//
//  common.c
//  FastFlexANN
//
//  Created by Wesley on 3/17/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#include <stdio.h>
#include "../inc/common.h"

void randperm(int *input, int len){
    for (int i=0; i<len-1; i++) {
        int rnd_idx = rand()%(len-i)+i;
        assert(rnd_idx<len && rnd_idx>=0);
        int tmp = input[i];
        input[i] = input[rnd_idx];
        input[rnd_idx] = tmp;
    }
}

fflex_msg_t find_kth_abs_smallest_elem(fflex_msg_t *input, int len,int k){
    assert(k<=len && k>0);
    fflex_msg_t local_max;
    int idx;
    fflex_msg_t *temp = (fflex_msg_t *)malloc(sizeof(fflex_msg_t)*k);
    memcpy(temp, input, k*sizeof(fflex_msg_t));
    for(int i=0;i<k;i++){
        temp[i] = fabs(temp[i]);
    }
    
    find_max(temp, k, &local_max, &idx);
    for(int i=k;i<len;i++){
        if(fabs(input[i])<local_max){
            temp[idx] = fabs(input[i]);
            find_max(temp, k, &local_max, &idx);
        }
    }
    find_max(temp, k, &local_max, &idx);
    free(temp);
    return local_max;
}

void find_max(fflex_msg_t *input, int len, fflex_msg_t *maxvalue, int *idx){
    assert(len>0 && input);
    *idx = 0;
    *maxvalue = input[0];
    for(int i=1;i<len;i++){
        if(input[i] > *maxvalue){
            *idx = i;
            *maxvalue = input[i];
        }
    }
}

int find_max_int(int *input, int len){
    assert(len>0 && input);
    int maxvalue = input[0];
    for(int i=1;i<len;i++){
        if(input[i] > maxvalue){
            maxvalue = input[i];
        }
    }
    return maxvalue;
}

int is_file_exist(char *file){
    FILE *fp = fopen(file, "r");
    if (fp){
        fclose(fp);
        return 1;
    }
    return 0;
}