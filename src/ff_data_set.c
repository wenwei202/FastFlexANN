//
//  ff_data_set.c
//  FastFlexANN
//
//  Created by Wesley on 3/11/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#include <stdio.h>
#include "../inc/ff_data_set.h"

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
                                      char * database){
    struct data_set_t * res = (struct data_set_t *) malloc(sizeof(struct data_set_t));
    res->training_file = fopen(train_file, "r");
    res->testing_file = fopen(test_file, "r");
    res->weight_file = fopen(weight_file, "rb");//always open

    res->log_file = fopen(log_file, "a+");
    time_t current_time = time(NULL);
    if (res->log_file) {
        fprintf(res->log_file, "/**************************************************************************\\\n");
        fprintf(res->log_file, "%s",asctime(gmtime(&current_time)));
        fprintf(res->log_file, "****************************************************************************\n");
    }else{
        printf("fail to open log file\n");
        exit(-1);
    }
    if (!res->training_file) {
        fprintf(res->log_file, "fail to open %s\n",train_file);
    }
    if (!res->testing_file) {
        fprintf(res->log_file, "fail to open %s\n",test_file);
    }
    
    res->training_features = NULL;
    res->training_labels = NULL;
    res->testing_features = NULL;
    res->testing_labels = NULL;
    res->weight_array = NULL;
    
    res->feature_num = 0;
    res->class_num = 0;
    res->training_sample_num = 0;
    res->testing_sample_num = 0;
    res->cur_epoch = 0;
    res->weight_num = 0;
    res->validate_sample_num = 0;
    res->training_sample_idx = NULL;
    
    if (0==strcmp("MNIST", database)
        || 0==strcmp("mnist", database)
        || 0==strcmp("TEXT", database)
        || 0==strcmp("text", database)) {
        //read data from mnist/txt format
        if (res->training_file) {
            fscanf(res->training_file, "%d %d %d\n",&res->feature_num,&res->class_num,&res->training_sample_num);
            res->training_features = (fflex_msg_t *) malloc(sizeof(fflex_msg_t)*res->feature_num*res->training_sample_num);
            res->training_labels = (fflex_class_t *) malloc(sizeof(fflex_class_t)*res->training_sample_num);
            for (int i=0; i<res->training_sample_num; i++) {
                for (int j=0; j<res->feature_num; j++) {
                    fscanf(res->training_file,"%lf ", &res->training_features[i*res->feature_num+j]);
                }
                fscanf(res->training_file,"%d\n", &res->training_labels[i]);
            }
        }
        
        if (res->testing_file) {
            fscanf(res->testing_file, "%d %d %d\n",&res->feature_num,&res->class_num,&res->testing_sample_num);
            res->testing_features = (fflex_msg_t *) malloc(sizeof(fflex_msg_t)*res->feature_num*res->testing_sample_num);
            res->testing_labels = (fflex_class_t *) malloc(sizeof(fflex_class_t)*res->testing_sample_num);
            for (int i=0; i<res->testing_sample_num; i++) {
                for (int j=0; j<res->feature_num; j++) {
                    fscanf(res->testing_file,"%lf ", &res->testing_features[i*res->feature_num+j]);
                }
                fscanf(res->testing_file,"%d\n", &res->testing_labels[i]);
            }
        }
    }else if ( 0==strcmp("CIFAR-10", database)
              || 0==strcmp("cifar-10", database)){
        /*
        //read cifar data from bin file
        int image_w = 32;
        int image_h = 32;
        int image_channel = 3;
        uint8 label_features[1+image_w*image_h*image_channel];
        if (res->training_file) {
            
            res->feature_num = image_w*image_h*image_channel;
            res->class_num = 10;
            res->training_sample_num = 50000;
            res->training_features = (fflex_msg_t *) malloc(sizeof(fflex_msg_t)*res->feature_num*res->training_sample_num);
            res->training_labels = (fflex_class_t *) malloc(sizeof(fflex_class_t)*res->training_sample_num);
            
            
            for (int i=0; i<res->training_sample_num; i++) {
                fread(label_features, sizeof(uint8), res->feature_num+1, res->training_file);
                res->training_labels[i] = label_features[0];
                for (int j=0; j<res->feature_num; j++) {
                    res->training_features[i*res->feature_num+j] = label_features[1+j]/127.5-1;
                }
            }
        }
        
        if (res->testing_file) {
            //fscanf(res->testing_file, "%d %d %d\n",&res->feature_num,&res->class_num,&res->testing_sample_num);
            res->feature_num = image_w*image_h*image_channel;
            res->class_num = 10;
            res->testing_sample_num = 10000;
            res->testing_features = (fflex_msg_t *) malloc(sizeof(fflex_msg_t)*res->feature_num*res->testing_sample_num);
            res->testing_labels = (fflex_class_t *) malloc(sizeof(fflex_class_t)*res->testing_sample_num);
            for (int i=0; i<res->testing_sample_num; i++) {
                
                fread(label_features, sizeof(uint8), res->feature_num+1, res->testing_file);
                res->testing_labels[i] = label_features[0];
                for (int j=0; j<res->feature_num; j++) {
                    res->testing_features[i*res->feature_num+j] = label_features[1+j]/127.5-1;
                }
                
            }
        }*/
        printf("bin format isn't supported! Please use text to format dataset \n");
        exit(-1);
    }
    
    
    res->validate_sample_num = res->training_sample_num/6;
    //randomly pick training samples
    res->training_sample_idx = (int *)malloc(sizeof(int)*res->training_sample_num);
    for (int i=0; i<res->training_sample_num; i++) {
        res->training_sample_idx[i] = i;//initially, it is in order
    }
    
    //read trained weights
    if (res->weight_file) {
        fread(&res->cur_epoch, sizeof(int), 1, res->weight_file);
        fread(&res->weight_num, sizeof(int), 1, res->weight_file);
        res->weight_array = (fflex_msg_t *)malloc(sizeof(fflex_msg_t)*res->weight_num);
        fread(res->weight_array, sizeof(fflex_msg_t), res->weight_num, res->weight_file);
        for (int i=0; i<res->weight_num; i++) {
            if (res->weight_array[i]>INT_MAX-2) {//deleted weights are valued as INT_MAX
                res->weight_array[i] = 0;
            }
        }
    }
    return res;
}


/**
 * @function release file resources
 */
void ff_close_data_files(struct data_set_t *data){
    
    if (data->training_file) fclose(data->training_file);
    if (data->testing_file) fclose(data->testing_file);
    if (data->weight_file) fclose(data->weight_file);
    if (data->log_file) fclose(data->log_file);
    
    if (data->training_features) {
        free(data->training_features);
    }
    if (data->training_labels) {
        free(data->training_labels);
    }
    if (data->testing_features) {
        free(data->testing_features);
    }
    if (data->testing_labels) {
        free(data->testing_labels);
    }
    if (data->weight_array) {
        free(data->weight_array);
    }
    
    data->training_features = NULL;
    data->training_labels = NULL;
    data->testing_features = NULL;
    data->testing_labels = NULL;
    data->weight_array = NULL;
    
    data->feature_num = 0;
    data->class_num = 0;
    data->training_sample_num = 0;
    data->testing_sample_num = 0;
    data->cur_epoch = 0;
    data->weight_num = 0;
    data->validate_sample_num = 0;
    if (data->training_sample_idx) {
        free(data->training_sample_idx);
        data->training_sample_idx = NULL;
    }
    
    free(data);
    data = NULL;
}

/**
 * @function train network with specifited parameter configuration
 * @param nn: the neural network;
 * @param config: parameter configuration
 */
void ff_train_on_file(struct ff_nn_t *nn,
                      struct parameters_conf_t config
                      ){
    
    char *training_file = config.train_file;
    char *testing_file = config.test_file;
    fflex_msg_t learning_rate = config.learning_rate;
    char *database = config.database;
    
    if (!nn || !training_file) {
        return;
    }
    
    //obtain corresponding weight file name
    char weight_filename[1000] = {'\0'};
    char str_tmp[1000]={'\0'};
    sprintf(weight_filename, "%s_weights_[",database);
    for (int i=0; i<nn->layer_num-1; i++) {
        sprintf(str_tmp, "%d_",nn->neuron_each_layer[i]);
        strcat(weight_filename, str_tmp);
    }
    sprintf(str_tmp, "%d]",nn->neuron_each_layer[nn->layer_num-1]);
    strcat(weight_filename, str_tmp);
    sprintf(str_tmp, "_%d_%d_%f_%f_%d",nn->connection_num,nn->userdata,learning_rate,config.pruning_rate,config.normalized);
    strcat(weight_filename, str_tmp);
    
    //open dataset
    printf("loading files...\n");
    char logfilename[200] = {'\0'};
    sprintf(logfilename, "%s.log",database);
    struct data_set_t * dataset = NULL;
    if(is_file_exist(config.weight_file)){
        dataset = ff_open_data_files(training_file,testing_file,config.weight_file,logfilename,database);
        if(dataset->weight_num!=nn->connection_num){
            fprintf(dataset->log_file, "network file (%d) doesn't match with weight file (%d)\n exited!\n",nn->connection_num,dataset->weight_num);
            exit(-1);
        }
    }else{
        dataset = ff_open_data_files(training_file,testing_file,weight_filename,logfilename,database);
    }
    
    printf("files loaded...\n");
    
    dataset->weight_num = nn->connection_num;
    char log_info[1000]={'\0'};
    int i=0;
    while (weight_filename[i]) {
        if (weight_filename[i]=='_') {
            log_info[i]=' ';
        }else{
            log_info[i]=weight_filename[i];
        };
        i++;
    }
    fprintf(dataset->log_file, "[parameter config: \n\t\t%s\n",log_info);
    fprintf(dataset->log_file, "network topology is read from %s\n",nn->network_file);
    
    //initialize weights
    if (dataset->weight_array ) {
        free(nn->weight_array);//warning
        fprintf(dataset->log_file, "weights loaded...\n");
        nn->weight_array = dataset->weight_array;
    }
    
    fprintf(dataset->log_file, "%d epochs are escaped since recovered weights...\n",dataset->cur_epoch);

    //training, validating, pruning and testing, previous epochs recovered from saved files are escaped
    int old_epochs = dataset->cur_epoch;
    for (int epoch_i=old_epochs+1; epoch_i<=config.epochs; epoch_i++) {
        randperm(dataset->training_sample_idx, dataset->training_sample_num); //randomly permutate in each epoch
        
        for (int j=0; j<dataset->training_sample_num-dataset->validate_sample_num; j++) {
            if ((j%2000)==0) {
                printf("%d\n",(epoch_i-1)*(dataset->training_sample_num-dataset->validate_sample_num)+j);
            }
            //training
            ff_nn_forward_prop(nn, dataset->training_features+dataset->training_sample_idx[j]*dataset->feature_num,TAN_SIGMOID);
            ff_nn_back_prop(nn, dataset->training_labels[dataset->training_sample_idx[j]],TAN_SIGMOID);
            ff_nn_update_weights(nn, learning_rate);

        }
        
        if (dataset->testing_file && dataset->testing_sample_num>0) {//basic check
            //evaluate test error
            float test_error_rate = ff_test_on_data(nn,dataset);
            //evaluate validation error
            float validate_error_rate = ff_validate_on_data(nn,dataset);
            fprintf(dataset->log_file, "[epoch %5d]: test(validate) error is %f(%f)\n",epoch_i,test_error_rate,validate_error_rate);
            
            //sparsify connections if validating error is less than 5%
            if (validate_error_rate<0.05 && round(config.pruning_rate*nn->left_connection_num)>0) {
#ifdef DEBUG
                printf("******************************** before *****************************\n");
                ff_nn_print_current_state(nn);
#endif
                dataset->cur_epoch = epoch_i;
                
                //save net befored pruning
                ff_save_network(nn,config,dataset);
                
                //prune connections whose weights are among smallest ones
                ff_nn_prune_connections(nn, STATIC,config.pruning_rate);
                
                fprintf(dataset->log_file, "[epoch %5d]: %8d connections are left [STATIC]\n",epoch_i,nn->left_connection_num);
#ifdef DEBUG
                printf("******************************** after *****************************\n");
                ff_nn_print_current_state(nn);
#endif
            }
        }
    }
    
    time_t current_time = time(NULL);
    fprintf(dataset->log_file, "%s",asctime(gmtime(&current_time)));
    fprintf(dataset->log_file, "\\--------------------------------------------------------------------------/\n\n\n");
    
    //save weights
    if (dataset->weight_file) {
        fclose(dataset->weight_file);//close read mode FILE
        dataset->weight_file = NULL;
    }
    if (config.epochs > old_epochs && (nn->connection_num==nn->left_connection_num)) {
        dataset->weight_file = fopen(weight_filename,"wb");
        dataset->cur_epoch = config.epochs;
        fwrite(&dataset->cur_epoch, sizeof(int), 1, dataset->weight_file);
        fwrite(&dataset->weight_num, sizeof(int), 1, dataset->weight_file);
        fwrite(nn->weight_array, sizeof(fflex_msg_t), dataset->weight_num, dataset->weight_file);
        
        ff_close_data_files(dataset);
        nn->weight_array = NULL;
        nn->connection_num = 0;
        nn->left_connection_num = 0;
    }
}

/**
 * @function test network
 * @param nn: the neural network;
 * @param data: data for testing
 * @param testing error
 */
float ff_test_on_data(struct ff_nn_t *nn,
                     struct data_set_t * data){
    fflex_msg_t maxvalue = -1;
    fflex_class_t max_class = -1;
    int total_error = 0;
    for (int i=0; i<data->testing_sample_num; i++) {
        //forward prop
        ff_nn_forward_prop(nn, data->testing_features+i*data->feature_num,TAN_SIGMOID);

        //find the maximum class
        int idx = nn->neuron_num - nn->neuron_each_layer[nn->layer_num-1];
        maxvalue = nn->neuron_array[idx].activation;
        max_class = 0;
        for (int j=idx+1; j<nn->neuron_num-1; j++) {
            if (nn->neuron_array[j].activation>maxvalue) {
                maxvalue = nn->neuron_array[j].activation;
                max_class = j-idx;
            }
        }
        if (max_class!=data->testing_labels[i]) {
            total_error++;
        }
    }
    return (float)total_error/(float)data->testing_sample_num;
}


/**
 * @function load network from file
 * @param file: file name of network
 * @return struct of network
 */
struct ff_nn_t *ff_load_network(const char *file){
    if (!file) {
        return NULL;
    }
    FILE *fp = fopen(file, "r");
    if (!fp) {
        return NULL;
    }
    
    int layer_num;
    fscanf(fp, "%d ",&layer_num);
    int *layers = (int *)malloc(sizeof(int)*layer_num);
    for (int i=0; i<layer_num; i++) {
        fscanf(fp,"%d ",&layers[i]);
        layers[i] += 1;//bias
    }
    int connection_num = 0;
    int userdata = 0;
    fscanf(fp,"%d %d\n",&connection_num, &userdata);
    struct ff_nn_t *nn = ff_nn_malloc(layers, layer_num, connection_num);
    nn->network_file = file;
    nn->userdata = userdata;
    
    int (*pair)[2];
    pair = (int (*)[2]) malloc(sizeof(int (*)[2])*connection_num);
    for (int i=0; i<connection_num; i++) {
        fscanf(fp, "%d %d\n",&((*(pair+i))[0]),&((*(pair+i))[1]));
    }
    ff_nn_connect(nn, pair);
    
    free(pair);
    fclose(fp);
    return nn;
}

/**
 * @function save network to file
 */
void ff_save_network(struct ff_nn_t *nn,struct parameters_conf_t config,struct data_set_t *dataset){
    
    //obtain weight file name
    char weight_filename[1000] = {'\0'};
    char str_tmp[1000]={'\0'};
    sprintf(weight_filename, "%s_weights_[",config.database);
    for (int i=0; i<nn->layer_num-1; i++) {
        sprintf(str_tmp, "%d_",nn->neuron_each_layer[i]);
        strcat(weight_filename, str_tmp);
    }
    sprintf(str_tmp, "%d]",nn->neuron_each_layer[nn->layer_num-1]);
    strcat(weight_filename, str_tmp);
    sprintf(str_tmp, "_%d_%d_%f_%f_%d",nn->left_connection_num,nn->userdata,config.learning_rate,config.pruning_rate,config.normalized);
    strcat(weight_filename, str_tmp);
    
    char net_file [1000] = "pruned_";
    strcat(net_file, weight_filename);
    strcat(net_file, ".txt");
    FILE * net_fp = fopen(net_file, "w");
    FILE * weight_fp = fopen(weight_filename, "wb");
    
    fprintf(net_fp, "%d ",nn->layer_num);
    for (int i=0; i<nn->layer_num; i++) {
        fprintf(net_fp,"%d ",nn->neuron_each_layer[i]-1);
    }
    fprintf(net_fp,"%d %d\n",nn->left_connection_num, nn->userdata);
    
    fwrite(&dataset->cur_epoch, sizeof(int), 1, weight_fp);
    fwrite(&nn->left_connection_num, sizeof(int), 1, weight_fp);
    
    
    
    int _neuron_updated = nn->neuron_num-1;
    int j;
    struct neuron_t *cur_neuron;
    struct neuron_t * _neuron_array = nn->neuron_array;
    fflex_msg_t * _weight_array = nn->weight_array;
    
    for (int i=0; i<_neuron_updated; i++) {
        cur_neuron = &_neuron_array[i];
        for (j=0; j<cur_neuron->forward_connection_num; j++) {
            fprintf(net_fp, "%d %d\n",i,cur_neuron->forward_neuron_idx[j]);
            fwrite(_weight_array+cur_neuron->forward_weight_idx[j], sizeof(fflex_msg_t), 1, weight_fp);
        }
    }
    
    fclose(weight_fp);
    fclose(net_fp);
}


/**
 * @function validate network
 * @param nn: the neural network;
 * @param data: data for validate
 * @param validating error
 */
float ff_validate_on_data(struct ff_nn_t *nn,
                          struct data_set_t * data){
    fflex_msg_t maxvalue = -1;
    fflex_class_t max_class = -1;
    int total_error = 0;
    int startIdx = data->training_sample_num - data->validate_sample_num;
    for (int i=0; i<data->validate_sample_num; i++) {
        //forward prop
        ff_nn_forward_prop(nn, data->training_features+data->training_sample_idx[startIdx+i]*data->feature_num,TAN_SIGMOID);

        //find the maximum class
        int idx = nn->neuron_num - nn->neuron_each_layer[nn->layer_num-1];
        maxvalue = nn->neuron_array[idx].activation;
        max_class = 0;
        for (int j=idx+1; j<nn->neuron_num-1; j++) {
            if (nn->neuron_array[j].activation>maxvalue) {
                maxvalue = nn->neuron_array[j].activation;
                max_class = j-idx;
            }
        }
        
        if (max_class!=data->training_labels[data->training_sample_idx[startIdx+i]]) {
            total_error++;
        }
    }
    return (float)total_error/(float)data->validate_sample_num;
}