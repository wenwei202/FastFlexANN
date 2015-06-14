# FastFlexANN
FastFlexANN is a C implementation of sparse artificial neural networks.

1. What’s FastFlexANN
FastFlexANN is a C implementation of sparse artificial neural networks (Multi-Layer Perceptron, MLP). Its advantages are:
o	Compact and easy to be understood;
o	Written by stand C to be portable to all C-compatible platforms;
o	The connective topology (for both fully and sparsely connected MLP) between layers can be flexibly specified by net file;
o	Connections can be pruned/reduced during training to sparsify nets;
o	Samples can be easily formatted as train file and test file.

2. What’s in repository
./inc - header files
./src - source files
./matlab – matlab scripts to generate MNIST samples and net files
./docs - documents