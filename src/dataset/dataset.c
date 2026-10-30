#include "dataset.h"
#include <stdio.h>



Dataset *load_dataset(char* filename, size_t input_length, size_t output_length){
	FILE *f = fopen(filename, "rb");
	if(!f)
		return NULL;

	Dataset *d = malloc(sizeof(Dataset));

	// Check malloc
	if(!d){
		fclose(f);
		return NULL;
	}
	
	size_t dataset_length;
	// Read first 4 bytes (sizeof(size_t)) to see the length of the dataset
	if(1 != fread(&dataset_length, sizeof(size_t), 1, f)){
		free(d);
		fclose(f);
		return NULL;
	}

	d->dataset_length = dataset_length;
	d->input_length = input_length;
	d->output_length = output_length;
	
	d->input_data = malloc(sizeof(float*) * dataset_length);
	// Check malloc
	if(!d->input_data){
		fclose(f);
		free(d);
		return NULL;
	}
	
	// Check malloc
	d->output_data = malloc(sizeof(float*) * dataset_length);
	if(!d->output_data){
		fclose(f);
		free(d->input_data);
		free(d);

		return NULL;
	}


	for(size_t i = 0; i < dataset_length; i++){

		 d->input_data[i] = malloc(sizeof(float) * input_length);

		// Check malloc
		if(! d->input_data[i]){
			fclose(f);
			for(size_t j = 0; j < i; j++){
				free(d->input_data[j]);
				free(d->output_data[j]);
			}
			free(d->input_data);
			free(d->output_data);
			free(d);
			return NULL;
		}

		 d->output_data[i] = malloc(sizeof(float) * output_length);

		// Check malloc
		if(!d->output_data[i]){
			fclose(f);
			free(d->input_data[i]);
			for(size_t j = 0; j < i; j++){
				free(d->input_data[j]);
				free(d->output_data[j]);
			}
			free(d->input_data);
			free(d->output_data);
			free(d);
			return NULL;
		}

		
		float *input = d->input_data[i];
		float *output = d->output_data[i];

		// read firstly output_length * sizeof(float) bytes  for values of the output
		if(output_length != fread(output, sizeof(float), output_length, f)){
			fclose(f);
			free(input);
			free(output);
			for(size_t j = 0; j < i; j++){
				free(d->input_data[j]);
				free(d->output_data[j]);
			}
			free(d->input_data);
			free(d->output_data);
			free(d);
			return NULL;
		}

		// read  input_length * sizeof(float) bytes for values of the input
		if(input_length != fread(input, sizeof(float), input_length, f)){
			fclose(f);
			free(input);
			free(output);
			for(size_t j = 0; j < i; j++){
				free(d->input_data[j]);
				free(d->output_data[j]);
			}
			free(d->input_data);
			free(d->output_data);
			free(d);
			return NULL;	
		}
	}
	fclose(f);
	return d;
}


int create_dataset(size_t dataset_length, size_t input_length, size_t output_length, char *filename, float **input_data, float **output_data){
	if(!filename || !input_data || !output_data)
		return -1;

	FILE *f = fopen(filename, "wb");
	if(!f)
		return -1;

	if(1 != fwrite(&dataset_length, sizeof(size_t), 1, f)){
		fclose(f);
		return -1;
	}

	for(size_t i = 0; i < dataset_length; i++){
		if(output_length != fwrite(output_data[i], sizeof(float), output_length, f)){
			fclose(f);
			return -1;
		}

		if(input_length != fwrite(input_data[i], sizeof(float), input_length, f)){
			fclose(f);
			return -1;
		}
	}

	fclose(f);
	return 0;
}


void destroy_dataset(Dataset *d){
	if(!d)
		return;

	for(size_t i = 0; i < d->dataset_length; i++){
		free(d->input_data[i]);
		free(d->output_data[i]);
	}
	free(d->input_data);
	free(d->output_data);
	free(d);	
}

/*
int main(){
	printf("Loading dataset...\n");
	Dataset *d = load_dataset("dataset.bin", 28*28, 1);
	printf("Dataset loaded!\n");

	destroy_dataset(d);
}
*/
