//
//  main.c
//  FastFlexANN
//
//  Created by Wesley on 3/10/15.
//  Copyright (c) 2015 Wesley. All rights reserved.
//

#include <stdio.h>
#include "../inc/neuron_t.h"
#include "../inc/common.h"
#include "../inc/ff_nn.h"
#include "../inc/ff_data_set.h"

int main(int argc, const char * argv[]) {
    //srand(time(NULL));
    if (argc<10) {
        printf("only %d params\n",argc);
        printf("./main [database] [epochs] [learning rate] [pruning rate] [normalized] [net file] [weight file] [train file] [test file]\n");
        exit(-2);
    }

    //initialize parameters
    struct parameters_conf_t config;
    char database[100]={'\0'};
    strcpy(database, argv[1]);
    config.database = database;
    config.epochs = atoi(argv[2]);
    config.learning_rate = atof(argv[3]);
    config.pruning_rate = atof(argv[4]);
    config.normalized = atoi(argv[5]);//0/1
    config.net_file = argv[6];
    char weight_file[1000]={'\0'};
    strcpy(weight_file, argv[7]);
    char train_file[1000]={'\0'};
    strcpy(train_file, argv[8]);
    char test_file[1000]={'\0'};
    strcpy(test_file, argv[9]);
    config.train_file = train_file;
    config.test_file = test_file;
    config.weight_file = weight_file;
    
    //load network from file
    register struct ff_nn_t *nn = ff_load_network(config.net_file);
    //train, validate and test neural networks
    ff_train_on_file(nn, config);
    //resource release
    ff_nn_free(nn);
    
#ifdef DEBUG
    printf("\nDEBUG MODE!!!!!!!\n");
#endif
    return 0;
}
