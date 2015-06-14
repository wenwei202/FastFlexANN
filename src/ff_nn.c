//
//  ff_nn.c
//  FastFlexANN
//
//  Created by Wesley on 3/10/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#include <stdio.h>
#include "../inc/ff_nn.h"

/**
 * @function allocate struct for neural networks, allocated neurons include bias neurons
 * @param neuron_each_layer - the neuron # in every layer
 * @param layer_num - the number of layers including input, hidden and output layers
 * @param connection_num - the number of connections in the neural networks including those with bias neurons
 * @return neural network struct
 */
struct ff_nn_t * ff_nn_malloc(int *neuron_each_layer,int layer_num,int connection_num){
    struct ff_nn_t *res = (struct ff_nn_t *) malloc(sizeof(struct ff_nn_t));
    res->neuron_each_layer = neuron_each_layer;
    res->layer_num = layer_num;
    res->connection_num = connection_num;
    res->left_connection_num = connection_num;
    int counter = 0;
    for (int i=0; i<layer_num; i++) {
        counter += neuron_each_layer[i];
    }
    res->neuron_num = counter;
    res->neuron_array = (struct neuron_t *)malloc(sizeof(struct neuron_t)*res->neuron_num);
    memset(res->neuron_array, 0, sizeof(struct neuron_t)*res->neuron_num);
    res->weight_array = (fflex_msg_t *)malloc(sizeof(fflex_msg_t)*res->connection_num);
    
    for (int i=0; i<res->connection_num; i++) {
        res->weight_array[i] = (rand()%2000-1000)/(float)20000;
        //res->weight_array[i] = 0;
    }
    //initialize bias
    int bias_neuron_idx = -1;
    for (int i=0; i<res->layer_num; i++) {
        bias_neuron_idx += res->neuron_each_layer[i];
        res->neuron_array[bias_neuron_idx].activation = 1;
    }
    return res;
}

/**
 * @function release memory and file resource
 * @param nn: the neural network
 */
void ff_nn_free(struct ff_nn_t * nn){
    if (nn) {
        if (nn->neuron_array) {
            free(nn->neuron_array);
            nn->neuron_array = NULL;
        }
        nn->neuron_num = 0;
        
        if (nn->weight_array) {
            free(nn->weight_array);
            nn->weight_array = NULL;
        }
        nn->connection_num = 0;
        nn->left_connection_num = 0;
        
        if (nn->neuron_each_layer) {
            free(nn->neuron_each_layer);
            nn->neuron_each_layer = NULL;
        }
        nn->layer_num = 0;
        
        for (int i=0; i<nn->neuron_num; i++) {
            free(nn->neuron_array[i].back_neuron_idx);
            nn->neuron_array[i].back_neuron_idx = NULL;
            free(nn->neuron_array[i].back_weight_idx);
            nn->neuron_array[i].back_weight_idx = NULL;
            nn->neuron_array[i].back_connection_num = 0;
            
            free(nn->neuron_array[i].forward_neuron_idx);
            nn->neuron_array[i].forward_neuron_idx = NULL;
            free(nn->neuron_array[i].forward_weight_idx);
            nn->neuron_array[i].forward_weight_idx = NULL;
            nn->neuron_array[i].forward_connection_num = 0;
        }

    }
}

/**
 * @function generate a fully-connected topology
 * @param nn - the neural network with layer info
 * @param pair - the returned connective relationship
 */
void ff_nn_generate_full_connection(struct ff_nn_t *nn,int (*pair)[2]){
    int neuron_count = 0;
    int connection_idx = 0;
    for (int i=0; i<nn->layer_num-1; i++) {
        for (int j=0; j<nn->neuron_each_layer[i]; j++) {
            for (int k=0; k<nn->neuron_each_layer[i+1]-1; k++) {
                (*(pair+connection_idx))[0] = neuron_count+j;
                (*(pair+connection_idx))[1] = neuron_count+nn->neuron_each_layer[i]+k;
                connection_idx++;
            }
        }
        neuron_count += nn->neuron_each_layer[i];
    }
    
}

/**
 * @function connect neurons in neural net
 * @param pair - the pairwise connections
 * @param nn - the neural network
 * @return the connected net
 */
struct ff_nn_t *ff_nn_connect(struct ff_nn_t *nn, int (*pair)[2]){
    
    for (int i=0; i<nn->connection_num; i++) {
        assert((*(pair+i))[0]<(*(pair+i))[1]);
        nn->neuron_array[(*(pair+i))[0]].forward_connection_num++;
        nn->neuron_array[(*(pair+i))[1]].back_connection_num++;
    }
    for (int i=0; i<nn->neuron_num; i++) {
        int mallocsize = nn->neuron_array[i].back_connection_num;
        nn->neuron_array[i].back_neuron_idx = (int *)malloc(sizeof(int)*mallocsize);
        nn->neuron_array[i].back_weight_idx = (int *)malloc(sizeof(int)*mallocsize);
        mallocsize = nn->neuron_array[i].forward_connection_num;
        nn->neuron_array[i].forward_neuron_idx = (int *)malloc(sizeof(int)*mallocsize);
        nn->neuron_array[i].forward_weight_idx = (int *)malloc(sizeof(int)*mallocsize);
    }
    
    int *back_counter = (int *)malloc(sizeof(int)*nn->neuron_num);
    memset(back_counter, 0, sizeof(int)*nn->neuron_num);
    int *forward_counter = (int *)malloc(sizeof(int)*nn->neuron_num);
    memset(forward_counter, 0, sizeof(int)*nn->neuron_num);
    int back_idx = 0;
    int forward_idx = 0;
    int connection_idx = -1;
    for (int i=0; i<nn->connection_num/*nn->neuron_num*/; i++) {
            connection_idx++;//connection idx
            back_idx = (*(pair+connection_idx))[0];//back neuron idx
            forward_idx = (*(pair+connection_idx))[1];//forward neuron idx
            nn->neuron_array[back_idx].forward_neuron_idx[forward_counter[back_idx]] = forward_idx;
            nn->neuron_array[forward_idx].back_neuron_idx[ back_counter[forward_idx] ] = back_idx;
            nn->neuron_array[back_idx].forward_weight_idx[forward_counter[back_idx]] = connection_idx;
            nn->neuron_array[forward_idx].back_weight_idx[ back_counter[forward_idx] ] = connection_idx;
            back_counter[forward_idx]++;
            forward_counter[back_idx]++;
    }
    free(back_counter);
    free(forward_counter);
    return nn;
}

char * ff_nn_get_current_state(struct ff_nn_t *nn, char *buffer){
    buffer[0]='\0';
    char tmp_c [1000];
    for (int i=0; i<nn->neuron_num; i++) {
        struct neuron_t cur_neuron = nn->neuron_array[i];
        sprintf(tmp_c, "%-4d-th neuron: %-10.6f %-10.6f %-10.6f\n",i,cur_neuron.sum,cur_neuron.activation,cur_neuron.bp_msg);
        strcat(buffer, tmp_c);
        for (int j=0; j<cur_neuron.forward_connection_num; j++) {
            sprintf(tmp_c, "-> %-4d %-10.6f\n",
                    cur_neuron.forward_neuron_idx[j],
                    nn->weight_array[cur_neuron.forward_weight_idx[j]]);
            strcat(buffer, tmp_c);
        }
        for (int j=0; j<cur_neuron.back_connection_num; j++) {
            sprintf(tmp_c, "<- %-4d %-10.6f\n",
                    cur_neuron.back_neuron_idx[j],
                    nn->weight_array[cur_neuron.back_weight_idx[j]]);
            strcat(buffer, tmp_c);
        }
    }
    return buffer;
}

void ff_nn_print_current_state(struct ff_nn_t *nn){
    char tmp_c [1000];
    for (int i=0; i<nn->neuron_num; i++) {
        struct neuron_t cur_neuron = nn->neuron_array[i];
        printf("%-4d-th neuron: %-10.6f %-10.6f %-10.6f\n",i,cur_neuron.sum,cur_neuron.activation,cur_neuron.bp_msg);
        for (int j=0; j<cur_neuron.forward_connection_num; j++) {
            printf("-> %-4d %lf\n",
                    cur_neuron.forward_neuron_idx[j],
                    nn->weight_array[cur_neuron.forward_weight_idx[j]]);
        }
        for (int j=0; j<cur_neuron.back_connection_num; j++) {
            printf("<- %-4d %lf\n",
                    cur_neuron.back_neuron_idx[j],
                    nn->weight_array[cur_neuron.back_weight_idx[j]]);
        }
    }
}

/**
 * @function forward propagation
 * @param nn - the neural network
 * @param x - input vector for input layer
 * @param type - the activation enum for output layer
 */
void ff_nn_forward_prop(struct ff_nn_t *nn, fflex_msg_t *x,enum output_activation_func type){
    int neuron_idx = 0;
    register struct neuron_t * _neuron_array = nn->neuron_array;
    register fflex_msg_t * _weight_array = nn->weight_array;
    int first_layer_x_len = nn->neuron_each_layer[0]-1;
    
    //initialize input layers
    for (neuron_idx=0; neuron_idx<first_layer_x_len; neuron_idx++) {
        _neuron_array[neuron_idx].activation = x[neuron_idx];
    }
    neuron_idx++;//skeep bias neuron
    
    //all hidden layers
    register struct neuron_t *cur_neuron;
    register fflex_msg_t sum = 0;
    int _layer_num = nn->layer_num;
    
    int j,k;
    int cur_neuron_num;
    for (int layer=1; layer<_layer_num-1; layer++) {
        cur_neuron_num = nn->neuron_each_layer[layer]-1;
        for (j=0; j<cur_neuron_num; j++) {
            cur_neuron = &_neuron_array[neuron_idx++];
            sum = 0;
            for (k=0; k<cur_neuron->back_connection_num; k++) {
                sum += _weight_array[cur_neuron->back_weight_idx[k]]*_neuron_array[cur_neuron->back_neuron_idx[k]].activation;
            };
#ifdef DEBUG
            if (cur_neuron->back_connection_num <= 0) {
                printf("neuron %d 's back neurons are completedly pruned\n",neuron_idx-1);
            }
#endif
            cur_neuron->sum = sum;
            cur_neuron->activation = TANSIGMOID(sum);//1/(1+exp(-sum));
        }
        neuron_idx++;//skeep bias neuron
    }
    
    //output layer
    cur_neuron_num = nn->neuron_each_layer[_layer_num-1]-1;
    for (j=0; j<cur_neuron_num; j++) {
        cur_neuron = &_neuron_array[neuron_idx++];
        sum = 0;
        for (k=0; k<cur_neuron->back_connection_num; k++) {
            sum += _weight_array[cur_neuron->back_weight_idx[k]]*_neuron_array[cur_neuron->back_neuron_idx[k]].activation;
        };
#ifdef DEBUG
        if (cur_neuron->back_connection_num <= 0) {
            printf("neuron %d 's back neurons are completedly pruned",neuron_idx-1);
        }
#endif
        cur_neuron->sum = sum;
        
        
        if (SOFT_MAX==type) {
            cur_neuron->activation = exp(sum);
        }else{
            cur_neuron->activation = TANSIGMOID(sum);
        }
    }
    neuron_idx++;//skeep bias neuron
}


/**
 * @function back propagation
 * @param nn - the neural network
 * @param t - the label/target of current training sample. For n-class recoginition, t varies from 0 ~ n-1
 * @param type - the activation enum for output layer
 */
void ff_nn_back_prop(struct ff_nn_t *nn, fflex_class_t t,enum output_activation_func type){
    int i,j,k;
    int neuron_idx = nn->neuron_num-1;
    register fflex_msg_t sum = 0;
    register struct neuron_t * _neuron_array = nn->neuron_array;
    register fflex_msg_t * _weight_array = nn->weight_array;
    fflex_msg_t maxvalue;
    int maxclass;
    //initialize bp msg in the output layer
    int ih_neuron_num = nn->neuron_num - nn->neuron_each_layer[nn->layer_num-1];
    neuron_idx--;//skeep bias
    if (SOFT_MAX==type) {
        //case SOFT_MAX:
            for (i = nn->neuron_num-2;
                 i >= ih_neuron_num;
                 i--) {
                sum += _neuron_array[i].activation;
                neuron_idx--;
            }
            for (i = nn->neuron_num-2;
                 i >= ih_neuron_num;
                 i--) {
                _neuron_array[i].bp_msg = _neuron_array[i].activation/sum;
            }
            _neuron_array[ih_neuron_num+t].bp_msg = _neuron_array[ih_neuron_num+t].bp_msg - 1;
    }else{
        //default://tan
            for (i = nn->neuron_num-2;
                 i >= ih_neuron_num;
                 i--) {
                _neuron_array[i].bp_msg = (_neuron_array[i].activation+1)*DTANSIGMOID(_neuron_array[i].activation);
                neuron_idx--;
            }
            _neuron_array[ih_neuron_num+t].bp_msg = (_neuron_array[ih_neuron_num+t].activation-1)*DTANSIGMOID(_neuron_array[ih_neuron_num+t].activation);
        
    }
    
    
    //back prop of hidden layers
    int cur_neuron_num;
    register fflex_msg_t msg_reg;
    register struct neuron_t *cur_neuron;
    for (int layer=nn->layer_num-2; layer>0; layer--) {
        cur_neuron_num = nn->neuron_each_layer[layer]-1;
        neuron_idx--;//skeep bias neuron
        for (j=cur_neuron_num-1; j>=0; j--) {
            cur_neuron = &_neuron_array[neuron_idx--];
            sum = 0;
            for (k=0; k<cur_neuron->forward_connection_num; k++) {
                sum += _weight_array[cur_neuron->forward_weight_idx[k]]*_neuron_array[cur_neuron->forward_neuron_idx[k]].bp_msg;
            };
#ifdef DEBUG
            if (cur_neuron->forward_connection_num <= 0) {
                printf("neuron %d 's forward neurons are completedly pruned",neuron_idx+1);
            }
#endif
            msg_reg = DTANSIGMOID(cur_neuron->activation);//NOT SUM
            cur_neuron->bp_msg = msg_reg*sum;
        }
        
    }
}

/**
 * @function weight updating based on gradient
 * @param nn - the neural network
 * @param learning_rate - learning rate for weights
 */
void ff_nn_update_weights(struct ff_nn_t *nn, fflex_msg_t learning_rate){
    
    int _neuron_updated = nn->neuron_num-1;
    int j;
    //int connection_idx = 0;
    register fflex_msg_t inner_factor;
    register struct neuron_t *cur_neuron;
    register struct neuron_t * _neuron_array = nn->neuron_array;
    register fflex_msg_t * _weight_array = nn->weight_array;
    for (int i=0; i<_neuron_updated; i++) {
        cur_neuron = &_neuron_array[i];
        inner_factor = learning_rate*cur_neuron->activation;
        for (j=0; j<cur_neuron->forward_connection_num; j++) {
            _weight_array[cur_neuron->forward_weight_idx[j]] -= inner_factor*(_neuron_array[cur_neuron->forward_neuron_idx[j]].bp_msg);
        }
    }
}

/**
 * @function prune connections, connections (with smallest absolute weights) are pruned when validating accuracy is larger than 95%
 * @param nn - the neural network
 * @param method - only STATIC are supported
 * @param pruning_rate - the rate of pruning when validating accuracy is satisfied
 */
void ff_nn_prune_connections(struct ff_nn_t *nn, enum weight_prune_method method,float prune_rate){

    assert(prune_rate<1 && prune_rate>=-0.000001);
    fflex_msg_t weight_thre = 0;
    
    int _neuron_updated = nn->neuron_num-1;
    struct neuron_t *cur_neuron;
    struct neuron_t * _neuron_array = nn->neuron_array;
    fflex_msg_t * _weight_array = nn->weight_array;
    int max_forward_conn = find_max_int(nn->neuron_each_layer+1, nn->layer_num-1);
    int *prune_idx = (int *)malloc(sizeof(int)*max_forward_conn);//connections which should be pruned are stored here
    int prune_num = 0;
    int old_left_conn = nn->left_connection_num;
    int ii=0;
    memset(prune_idx, 0, sizeof(int)*max_forward_conn);
    switch(method){
            case STATIC:
                weight_thre = find_kth_abs_smallest_elem(nn->weight_array, nn->connection_num, round(nn->left_connection_num*prune_rate));
#ifdef DEBUG
printf("weight threshold: %lf\n",weight_thre);
            for(int kk=0;kk<nn->connection_num;kk++){
                printf("Marker %lf\n",_weight_array[kk]);
            }
#endif
                for (int i=0; i<_neuron_updated; i++) {
                    cur_neuron = &_neuron_array[i];
                    prune_num = 0;

                    for (int j=0; j<cur_neuron->forward_connection_num; j++) {
                        assert(fabs(_weight_array[cur_neuron->forward_weight_idx[j]])<INT_MAX-2);
                        if (10000*fabs(_weight_array[cur_neuron->forward_weight_idx[j]])<=10000*weight_thre) {//inaccuray may increase or decrease the expected number of pruned connections
                            prune_idx[prune_num++] = j;
                        }
                    }
#ifdef DEBUG
printf("pruned connections of neuron %d:\n",i);
for(int kk=0;kk<prune_num;kk++){
    printf("%d ",prune_idx[kk]);
}
printf("\n");
#endif
                    //pruning
                    ii = 0;
                    for (int bubble=cur_neuron->forward_connection_num-1;
                         bubble >= cur_neuron->forward_connection_num-prune_num;
                         bubble--) {
                            int to_prune_idx;
                            int temp_idx2 = cur_neuron->forward_weight_idx[bubble];
                        
                            if (fabs(_weight_array[temp_idx2])>weight_thre) {//inaccuray is tolerable
                                to_prune_idx = prune_idx[ii++];
                            }else{
                                to_prune_idx = bubble;
                            }
                            int next_neuron_idx = cur_neuron->forward_neuron_idx[to_prune_idx];
                            _weight_array[ cur_neuron->forward_weight_idx[to_prune_idx] ] = INT_MAX;//won't appear in the smallest values
                            if (to_prune_idx!=bubble) {
                                cur_neuron->forward_weight_idx[to_prune_idx] = temp_idx2;
                                cur_neuron->forward_neuron_idx[to_prune_idx] = cur_neuron->forward_neuron_idx[bubble];
                            }
                        
                            cur_neuron->forward_weight_idx[bubble] = 0;//useless connection
                            cur_neuron->forward_neuron_idx[bubble] = 0;
                        
                            //prune the backward connection of next layer
                            struct neuron_t *next_neuron = _neuron_array+next_neuron_idx;
                            int jj=0;
                            for (jj=0; jj<next_neuron->back_connection_num; jj++) {
                                //find the current neuron and prune the connection between them
                                if (_neuron_array+next_neuron->back_neuron_idx[jj] == cur_neuron) {
                                    next_neuron->back_neuron_idx[jj]=next_neuron->back_neuron_idx[next_neuron->back_connection_num-1];
                                    next_neuron->back_weight_idx[jj]=next_neuron->back_weight_idx[next_neuron->back_connection_num-1];
                                    next_neuron->back_neuron_idx[next_neuron->back_connection_num-1] = 0;//useless connection
                                    next_neuron->back_weight_idx[next_neuron->back_connection_num-1] = 0;
                                    next_neuron->back_connection_num--;
                                    break;
                                }
                            }
                            assert(jj<=next_neuron->back_connection_num);
                        
                    }
                    cur_neuron->forward_connection_num -= prune_num;
                    nn->left_connection_num -= prune_num;
                }
#ifdef DEBUG
            printf("connections are pruned from %d to %d\n",old_left_conn,nn->left_connection_num);
#endif
                break;
            default:
                break;
    }
    
}