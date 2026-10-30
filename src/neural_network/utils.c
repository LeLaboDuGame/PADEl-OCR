#include "utils.h"
#include <math.h>

float random1m1(){
	return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}


void fill_random_array(float *arr, size_t n, size_t fan_in){
	float scale = sqrtf(2.0f / (float)fan_in);
	for(size_t i = 0; i < n; i++){
		// tirage uniforme dans [-1, 1] mis à l'échelle
		float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
		arr[i] = r * scale;
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
