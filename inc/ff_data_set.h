//
//  ff_data_set.h
//  FastFlexANN
//
//  Created by Wesley on 3/11/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#ifndef FastFlexANN_ff_data_set_h
#define FastFlexANN_ff_data_set_h
#include "common.h"
#include "ff_nn.h"

/**
 * @struct store parameters and configuration
 */
struct data_set_t {
    
    //file pointers
    FILE * training_file; //point to training data
    FILE * testing_file;  //point to testing data
    FILE * weight_file;   //point to stored weight data
    FILE * log_file;      //point to log file
    
    //training samples, testing samples and weights
    fflex_msg_t *training_features;
    fflex_class_t *training_labels;
    fflex_msg_t *testing_features;
    fflex_class_t *testing_labels;
    fflex_msg_t * weight_array;
    
    int feature_num; //feature/input dimension
    int class_num;   //number of classes
    int training_sample_num;
    int testing_sample_num;
    int cur_epoch; //current epoch
    int weight_num;//length of weight_array
    int validate_sample_num;//number of samples used for validating
    int *training_sample_idx;//random order of indices for training_features, first (training_sample_num - validate_sample_num) samples are for training, and last validate_sample_num ones are for validating
};

/**
 * @struct store parameters and configuration
 */
struct parameters_conf_t{
    char *database; //the database for training and testing (Supported formats: general text, CIFAR)
    int epochs; //epochs to train&test the entire database
    float learning_rate; //gradient learning rate
    float pruning_rate; //pruning rate (percentage) of remaining connections (pruning_rate can be 0)
    int normalized;// useless in this version
    const char * net_file; //file name of network
    char *train_file; //file name of training data
    char *test_file; //file name of testing data
    char *weight_file;//file name of saved weights
};

/**
 * @function train network with specifited parameter configuration
 * @param nn: the neural network; 
 * @param config: parameter configuration
 */
void ff_train_on_file(struct ff_nn_t *nn,
                      struct parameters_conf_t config
                      );

/**
 * @function test network
 * @param nn: the neural network;
 * @param data: data for testing
 * @param testing error
 */
float ff_test_on_data(struct ff_nn_t *nn,
                     struct data_set_t * data);

/**
 * @function validate network
 * @param nn: the neural network;
 * @param data: data for validate
 * @param validating error
 */
float ff_validate_on_data(struct ff_nn_t *nn,
                      struct data_set_t * data);

/**
 * @function read dataset
 * @param database: dataset for ML (MNIST, txt ...)
 * @return the dataset struct
 */
struct data_set_t *ff_open_data_files(
                                      char *train_file,
                                      char * test_file,
                                      char * weight_file,
                                      char * log_file,
                                      char * database);
/**
 * @function release file resources
 */
void ff_close_data_files(struct data_set_t *);

/**
 * @function load network from file
 * @param file: file name of network
 * @return struct of network
 */
struct ff_nn_t * ff_load_network(const char *file);

/**
 * @function save network to file
 */
void ff_save_network(struct ff_nn_t *nn,struct parameters_conf_t config, struct data_set_t *dataset);
#endif
