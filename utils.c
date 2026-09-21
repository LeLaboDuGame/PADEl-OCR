#include "utils.h"

float random1m1(){
	return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}


void fill_random_array(float *array, size_t len){
	for(size_t i = 0; i < len; i++){
		array[i] = random1m1();
	}
}


void print_vector(float *A, size_t len)
{
    printf("[ ");
    for (size_t i = 0; i < len; i++) {
        printf("%f ", A[i]);
    }
    printf("]\n");
}
