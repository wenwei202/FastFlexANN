//
//  ff_nn.h
//  FastFlexANN
//
//  Created by Wesley on 3/10/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#ifndef FastFlexANN_ff_nn_h
#define FastFlexANN_ff_nn_h

#include "neuron_t.h"
#include "common.h"

/**
 * @struct data structure for neural networks
 */
struct ff_nn_t{
    /**
     * neuron array:
     * neurons may not be arrayed in layer order
     * a bias neuron in every layer should also be included
     * the bias neuron in the last layer is useless except for programming convenience
     */
    struct neuron_t * neuron_array;
    
    /**
     * all weight array on connections
     */
    fflex_msg_t * weight_array;
    
    //array of neuron number in every layer including the bias neuron
    int *neuron_each_layer;
    
    //the number of layers, including input, hidden and output layers
    int layer_num;
    
    //the number of connections in a standard fully-connected NN
    int connection_num;
    
    //the number of connections in a sparsified NN
    int left_connection_num;
    
    //the number of neurons in all layers (neuron_each_layer)
    int neuron_num;
    
    //the file name of read network
    const char *network_file;
    
    int userdata;
    
    
};

/**
 * @function allocate struct for neural networks, allocated neurons include bias neurons
 * @param neuron_each_layer - the neuron # in every layer
 * @param layer_num - the number of layers including input, hidden and output layers
 * @param connection_num - the number of connections in the neural networks including those with bias neurons
 * @return neural network struct
 */
struct ff_nn_t * ff_nn_malloc(int *neuron_each_layer,int layer_num,int connection_num);

/**
 * @function release memory and file resource
 * @param nn: the neural network
 */
void ff_nn_free(struct ff_nn_t * nn);

/**
 * @function connect neurons in neural net 
 * @param pair - the pairwise connections
 * @param nn - the neural network
 * @return the connected net
 */
struct ff_nn_t *ff_nn_connect(struct ff_nn_t *nn, int (*pair)[2]);

/**
 * @function return/print info about weight values, neuron states and connective relationship, only used for debugging
 * @bug large neural network can result memory overflow
 */
char * ff_nn_get_current_state(struct ff_nn_t *nn, char *buffer);
void ff_nn_print_current_state(struct ff_nn_t *nn);

/**
 * @function generate a fully-connected topology
 * @param nn - the neural network with layer info
 * @param pair - the returned connective relationship
 */
void ff_nn_generate_full_connection(struct ff_nn_t *nn,int (*pair)[2]);

/**
 * @function forward propagation
 * @param nn - the neural network
 * @param x - input vector for input layer
 * @param type - the activation enum for output layer
 */
void ff_nn_forward_prop(struct ff_nn_t *nn, fflex_msg_t *x, enum output_activation_func type);

/**
 * @function back propagation
 * @param nn - the neural network
 * @param t - the label/target of current training sample. For n-class recoginition, t varies from 0 ~ n-1
 * @param type - the activation enum for output layer
 */
void ff_nn_back_prop(struct ff_nn_t *nn, fflex_class_t t,enum output_activation_func type);

/**
 * @function weight updating based on gradient
 * @param nn - the neural network
 * @param learning_rate - learning rate for weights
 */
void ff_nn_update_weights(struct ff_nn_t *nn,fflex_msg_t learning_rate);

/**
 * @function prune connections, connections (with smallest absolute weights) are pruned when validating accuracy is larger than 95%
 * @param nn - the neural network
 * @param method - only STATIC are supported
 * @param pruning_rate - the rate of pruning when validating accuracy is satisfied
 */
void ff_nn_prune_connections(struct ff_nn_t *nn, enum weight_prune_method method, float pruning_rate);
#endif
