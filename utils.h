#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>


// Get a random network between -1 and +1
float random1m1();

// Fill an array with random values
void fill_random_array(float *arr, size_t n, size_t fan_in);
// Print a vector.
void print_vector(float *A, size_t len);


#endif
