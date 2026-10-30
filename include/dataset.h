#ifndef DATASET_H
#define DATASET_H
#include <stdlib.h>


// Dataset
typedef struct{
	float **input_data;
	float **output_data;
	size_t dataset_length;
	size_t input_length;
	size_t output_length;
} Dataset;

Dataset *load_dataset(char* filename, size_t input_length, size_t output_length);
int create_dataset(size_t dataset_length, size_t input_length, size_t output_length, char *filename, float **input_data, float **output_data);
void destroy_dataset(Dataset *d);

#endif
