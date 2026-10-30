#ifndef NN_H
#define NN_H

#include <stdlib.h>

// Activation type
typedef struct {
  void (*forward)(float *Z, float *A, size_t n);      // ex: relu, sigmoid, softmax
  void (*backward)(float *dA, float *Z, float *dZ, size_t n); // * dZ
} Activation;


// Loss type
typedef struct {
  float (*forward)(float *Ypred, float *T, size_t n);
  void (*backward)(float *Ypred, float *T, float *dA, size_t n);
} Loss;

// Layer struct
struct layer {
	float *bias; // bias parameter
	Activation activation;
	size_t in; // Number of neural
	size_t out; // Number of neural to the next layer
	float *weights; // weights parameters W[o*in+i]
};

// Neural Network struct
struct neural_network {
	struct layer *layers; // Neura layers
	Loss loss;
	size_t output_neural_len; // Numbers of neural at the output
	size_t len; // Number of layers
};


// Create a neural_network struct
struct neural_network *create_nn(
	size_t layers[], // Hiden layers
	Activation activations[],
	Loss loss,
	size_t n_layers // Length of hiden layers, it is len(nn.layers)
);

// Cache
typedef struct {
	float *A;
	float *Z;	
} Cache;


// Create a Cache
Cache *create_caches(struct neural_network *nn);

// Destroy the Caches
void destroy_caches(Cache *c, size_t len);

// Gradient
typedef struct {
	float *dA;
	float *dZ;	
} Grad;

// Create a Cache
Grad *create_grads(struct neural_network *nn);

// Destroy the Caches
void destroy_grads(Grad *g, size_t len);


// Print a summary of the neural network
void nn_summary(struct neural_network *nn);

// Destory/Free the neural_network struct
void destroy_nn(struct neural_network *nn);

// Forward propagation
void forward(struct neural_network *nn, float* X, Cache *caches);

// Backward propagation
void backward(struct neural_network *nn, Cache *caches, Grad *grads, float* T, float lr);

void train(
  struct neural_network *nn,
  Cache *caches, Grad *grads,
  float **Xs, float **Ts, // dataset: Input, True outpur
  size_t N,	// Number of exemples
  size_t epochs, // Number of epoch for the ai to be trained
  float lr // Learning rate
);

// Will save the model in a custom path
int save_model(struct neural_network *nn, char* filepath);

// Will load the model and return a new neural_network struct
struct neural_network *load_model(char* filepath, Loss loss, Activation activations[]);


#endif
