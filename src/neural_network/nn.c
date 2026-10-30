#include "nn.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "dataset.h"

// Verbose level
short VERBOSE = 2;

// Forward declarations of the concrete backward functions defined in the
// TESTING section below. Needed here so backward() can detect the
// softmax+CE pair for the numerically fused path (pointer comparison).
void dsoftmax(float *dA, float *Z, float *A, float *dZ, size_t n);
void dce(float *Ypred, float *T, float *dA, size_t n);


// Create a neural_network struct
struct neural_network *create_nn(
	size_t layers[], // Hiden layers
	Activation activations[],
	Loss loss,
	size_t n_layers // Length of hiden layers, it is len(nn.layers)
){

	// Malloc the neural_network struct
	struct neural_network *nn = malloc(sizeof(struct neural_network));
	// Verify that malloc succeed
	if(!nn)
		return NULL;
	nn->loss = loss;
	nn->output_neural_len = layers[n_layers - 1];

	// Malloc the layers
	nn->layers =  malloc(sizeof(struct layer) * (n_layers - 1));
	
	// Verify that malloc succeed
	if(!nn->layers){
		free(nn);
		return NULL;
	}
	nn->len = n_layers - 1;

	for(size_t i = 0; i < n_layers - 1; i++){
		struct layer *l = (nn->layers + i);
		l->in = layers[i]; 
		l->out = layers[i + 1];
		l->activation = activations[i];


		// malloc and fill bias
		l->bias = malloc(sizeof(float) * layers[i + 1]);

		// Verify that malloc succeed
		if(!l->bias){
			for(size_t j = 0; j < i; j++){
				free(nn->layers[j].weights);
				free(nn->layers[j].bias);
			}
			free(nn->layers);
			free(nn);
			return NULL;
		}
		fill_random_array(l->bias, layers[i + 1], 1);
		
		// Malloc weights array
		l->weights = malloc(sizeof(float) * layers[i+1] * layers[i]);

		// Verify that malloc succeed
		if(!l->weights){
			free(l->bias);
			for(size_t j = 0; j < i; j++){
				free(nn->layers[j].weights);
				free(nn->layers[j].bias);
			}
			free(nn->layers);
			free(nn);
			return NULL;
		}
				
		fill_random_array(l->weights, layers[i+1] * layers[i], layers[i]);
	}

	
	return nn;
}




// Print a summary of the neural network
void nn_summary(struct neural_network *nn){
	printf("Neural Network summary:\nNumber of neural layers: %i\n\nLayers:\n",(int) nn->len + 1);
	
	for(size_t i = 0; i < nn->len; i++){
		struct layer *l = (nn->layers + i);
		printf("L%i - Neural: %i->%i\n", (int) i, (int) l->in, (int) l->out);
	}
	printf("Output layer: Neural: %i\n",(int) nn->output_neural_len);
	int total_param = 0;
	for(size_t i = 0; i < nn->len; i++){
		struct layer *l = (nn->layers + i);
		total_param += l->in * l->out;
		total_param += l->out;
	}
	printf("\nTotal parameters: %i - Memory: %ioctets\n", total_param, total_param * (int)sizeof(float));
	printf("Summary end!\n");
}


// Destory/Free the neural_network struct
void destroy_nn(struct neural_network *nn){
	size_t len = nn->len;
	for(size_t i = 0; i < len; i++){
		struct layer *l = (nn->layers + i);
		free(l->weights);
		free(l->bias);
	}
	free(nn->layers);
	free(nn);
}

// Create a Cache
Cache *create_caches(struct neural_network *nn){
	Cache *caches = malloc(sizeof(Cache) * (nn->len + 1));

	// Verrify that malloc succeed
	if(!caches)
		return NULL;

	caches[0].A = malloc(sizeof(float) * nn->layers[0].in);

	// Verrify that malloc succeed
	if(!caches[0].A){
		free(caches);
		return NULL;
	}
	
	caches[0].Z = NULL; // as the first layer dont have Z init it to NULL
	
	for(size_t i = 0; i < nn->len; i++){
		size_t out = nn->layers[i].out;
		
		caches[i + 1].A = malloc(sizeof(float) * out);

		// Verrify that malloc succeed
		if(!caches[i + 1].A){
			free(caches[0].A);
			for(size_t j = 1; j <= i; j++){
				free(caches[j].A);
				free(caches[j].Z);
			}
			free(caches);
			return NULL;
		}
		
		caches[i + 1].Z = malloc(sizeof(float) * out);
		if(!caches[i + 1].Z){
			free(caches[0].A);
			free(caches[i + 1].A);
			for(size_t j = 1; j <= i; j++){
				free(caches[j].A);
				free(caches[j].Z);

			}
			free(caches);
			return NULL;
		}
		
	}
	return caches;
}



// Destroy the Caches
void destroy_caches(Cache *c, size_t len){
	free(c[0].A);
	for(size_t i = 1; i < len; i++){
		free(c[i].A);
		free(c[i].Z);
	}
	free(c);
}




// Create a Gradients
Grad *create_grads(struct neural_network *nn){
	Grad *g = malloc(sizeof(Grad) * (nn->len + 1));

	// Verrify that malloc succeed
	if(!g)
		return NULL;

	g[0].dA = malloc(sizeof(float) * nn->layers[0].in);

	// Verrify that malloc succeed
	if(!g[0].dA){
		free(g);
		return NULL;
	}
	
	g[0].dZ = NULL; // as the first layer dont have Z init it to NULL
	
	for(size_t i = 0; i < nn->len; i++){
		size_t out = nn->layers[i].out;
		
		g[i + 1].dA = malloc(sizeof(float) * out);

		// Verrify that malloc succeed
		if(!g[i + 1].dA){
			free(g[0].dA);
			for(size_t j = 1; j <= i; j++){
				free(g[j].dA);
				free(g[j].dZ);
			}
			free(g);
			return NULL;
		}
		
		g[i + 1].dZ = malloc(sizeof(float) * out);
		if(!g[i + 1].dZ){
			free(g[0].dA);
			free(g[i + 1].dA);
			for(size_t j = 1; j <= i; j++){
				free(g[j].dA);
				free(g[j].dZ);

			}
			free(g);
			return NULL;
		}
		
	}
	return g;
}

// Destroy the Gradients
void destroy_grads(Grad *g, size_t len){
	free(g[0].dA);
	for(size_t i = 1; i < len; i++){
		free(g[i].dA);
		free(g[i].dZ);
	}
	free(g);
}


void forward(struct neural_network *nn, float* X, Cache *caches){
	for(size_t i = 0; i < nn->layers[0].in; i++){
		caches[0].A[i] = X[i];
	}

	for(size_t l = 0; l < nn->len; l++){
		struct layer *L = &nn->layers[l];
		float *Aprev = caches[l].A; // of sized L.in
		float *Zcur = caches[l+1].Z; // of sized L.out 
		float *Acur = caches[l+1].A; // of sized L.out

		for(size_t o = 0; o < L->out; o++){ // o = output
			float z = L->bias[o];
			for(size_t i = 0; i < L->in; i++){ // i = input
				z += L->weights[L->in*o + i] * Aprev[i]; // W[shift + index] * A[l-1]
			}
			Zcur[o] = z; // apply to Z[o]
		}

		// Apply the forward activation
		L->activation.forward(Zcur, Acur, L->out);
	}
}

void backward(struct neural_network *nn, Cache *caches, Grad *grads, float* T, float lr){
	float *Ypred = caches[nn->len].A;
	float *dA_last = grads[nn->len].dA;
	float *dZ_last = grads[nn->len].dZ;
	float *Z_last = caches[nn->len].Z;
	struct layer *L = &nn->layers[nn->len-1];

	nn->loss.backward(Ypred, T, dA_last, L->out);
	L->activation.backward(dA_last, Z_last, Ypred, dZ_last, L->out);

	// Premiere passe: on calcule tous les dZ sans toucher aux poids, pour que
	// la retropropagation utilise toujours les poids du step courant. Mettre a
	// jour la derniere couche avant de propager faisait diverger le reseau
	for(int l = (int)nn->len - 2; l >= 0; l--){
		struct layer *Lcur = &nn->layers[l];
		struct layer *Lnext = &nn->layers[l+1];
		for(size_t i = 0; i < Lnext->in; i++){
			float sum = 0;
			for (size_t k = 0; k < Lnext->out; k++){
				sum += Lnext->weights[k*Lnext->in + i] * grads[l+2].dZ[k];
			}
			grads[l+1].dA[i] = sum;
		}
		Lcur->activation.backward(grads[l + 1].dA, caches[l+1].Z, caches[l+1].A, grads[l+1].dZ, Lcur->out);
	}

	// Deuxieme passe: on applique les gradients
	for(size_t l = 0; l < nn->len; l++){
		struct layer *Lcur = &nn->layers[l];
		float *dZ = grads[l+1].dZ;
		float *Aprev = caches[l].A;
		for(size_t o = 0; o < Lcur->out; o++){
			float d_bias = lr * dZ[o];
			Lcur->bias[o] -= d_bias;
			for(size_t i = 0; i < Lcur->in; i++){
				Lcur->weights[o*Lcur->in+i] -= lr * dZ[o] * Aprev[i];
			}
		}
	}
}


void train(
  struct neural_network *nn,
  Cache *caches, Grad *grads,
  float **Xs, float **Ts, // dataset: Input, True outpur
  size_t N,	// Number of exemples
  size_t epochs, // Number of epoch for the ai to be trained
  float lr // Learning rate
){
	printf("Starting training over %i epochs on a %i dataset length...\n", (int)epochs, (int)N);
	float mean_loss = 0;
	for(size_t e = 0; e < epochs; e++){
		float loss_sum = 0;
		for(size_t n = 0; n < N; n++){
			forward(nn, Xs[n], caches);
			if(VERBOSE >= 1){
				loss_sum += nn->loss.forward(caches[nn->len].A, Ts[n], nn->layers[nn->len - 1].out);
			}
			backward(nn, caches, grads, Ts[n], lr);
		}
		if(VERBOSE >= 1){
			printf("Epochs n°%i/%i - Loss: %.10f - diff: %.10f\n", (int)e, (int)epochs, loss_sum / N, loss_sum / N - mean_loss);
			mean_loss = loss_sum / N;

		}
		
	}
	printf("Training finished !\n");
}


/// SAVE FILE PARTITION:
/// {len of the layer in bytes = sizeof(size_t)} | {Network Shape = sizeof(size_t) * len} |
/// {(len of weights in bytes = sizeof(float) * L[i].in * L[i].out) + (len of bias in bytes = sizeof(float) * L[i].out)} 
int save_model(struct neural_network *nn, char* filepath){
	if(!nn)
		return -1;
	FILE *file = fopen(filepath, "wb");
	if(!file){
		return -2;
	}
	// write len of the shape in bytes of size_t
	size_t len = nn->len + 1;
	if(1 != fwrite(&len, sizeof(size_t), 1, file)){
 		fclose(file);
 		return -2;
 	}
	
	
	// Write first layer size
	if(1 != fwrite(&nn->layers[0].in, sizeof(size_t), 1, file)){
		fclose(file);
		return -2;
	}
	
	// Write others layers size
	for(size_t i = 0; i < nn->len; i++){
		if(1 != fwrite(&nn->layers[i].out, sizeof(size_t), 1, file)){
			fclose(file);
			return -2;
		}
	}

	for(size_t j = 0; j < nn->len; j++){
		struct layer *L = &nn->layers[j];
		// Write weights of L[i]
		if(L->in * L->out != fwrite(L->weights, sizeof(float), L->in * L->out, file)){
			fclose(file);
			return -2;
		}

		// Write bias of L[i]
		if(L->out != fwrite(L->bias, sizeof(float), L->out, file)){
			fclose(file);
			return -2;
		}
		
	}
	
	
	fclose(file);
	return 0;
}

struct neural_network *load_model(char* filepath, Loss loss, Activation activations[]){
	

	FILE *file = fopen(filepath, "rb");
	if(!file)
		return NULL;

	size_t n_layers = 0;
	
	// Read the length of layers
	if(1 != fread(&n_layers, sizeof(size_t), 1, file)){
		fclose(file);
		return NULL;
	}
	size_t *layers = malloc(sizeof(size_t) * n_layers);
	// Verify that malloc succeed
	if(!layers){
		fclose(file);
		return NULL;
	}
	// Read layers shape
	if(n_layers != fread(layers, sizeof(size_t), n_layers, file)){
		fclose(file);
		free(layers);
		return NULL;
	}

	struct neural_network *nn = malloc(sizeof(struct neural_network));
	
	// Verify that malloc succeed
	if(!nn){
		fclose(file);
		free(layers);
		return NULL;
	}
	nn->loss = loss;

	
	nn->output_neural_len = layers[n_layers - 1];

	// Malloc the layers
	nn->layers =  malloc(sizeof(struct layer) * (n_layers - 1));
	
	// Verify that malloc succeed
	if(!nn->layers){
		free(nn);
		free(layers);
		fclose(file);
		return NULL;
	}
	
	nn->len = n_layers - 1;

	for(size_t i = 0; i < n_layers - 1; i++){
		struct layer *l = (nn->layers + i);
		l->in = layers[i]; 
		l->out = layers[i + 1];
		l->activation = activations[i];


		// Malloc weights array
		l->weights = malloc(sizeof(float) * layers[i+1] * layers[i]);

		// Verify that malloc succeed
		if(!l->weights){
			for(size_t j = 0; j < i; j++){
				free(nn->layers[j].weights);
				free(nn->layers[j].bias);
			}
			free(nn->layers);
			free(nn);
			
			free(layers);
			fclose(file);
			return NULL;
		}
		
		// Read and load weights from the file
		if(l->in * l->out != fread(l->weights, sizeof(float), l->in * l->out, file)){
			free(l->weights);
			for(size_t j = 0; j < i; j++){
				free(nn->layers[j].weights);
				free(nn->layers[j].bias);
			}
			free(nn->layers);
			free(nn);
			
			free(layers);
			fclose(file);
			return NULL;
		}
		
		// malloc and fill bias
		l->bias = malloc(sizeof(float) * layers[i + 1]);

		// Verify that malloc succeed
		if(!l->bias){
			free(l->weights);
			for(size_t j = 0; j < i; j++){
				free(nn->layers[j].weights);
				free(nn->layers[j].bias);
			}
			free(nn->layers);
			free(nn);
			
			free(layers);
			fclose(file);
			return NULL;
		}

		// Read and load bias from the file
		if(l->out != fread(l->bias, sizeof(float), l->out, file)){
			free(l->weights);
			free(l->bias);
			for(size_t j = 0; j < i; j++){
				free(nn->layers[j].weights);
				free(nn->layers[j].bias);
			}
			free(nn->layers);
			free(nn);
			
			free(layers);
			fclose(file);
			return NULL;
		}
	}
	free(layers);
	fclose(file);
	
	return nn;
}


void print_params(struct neural_network *nn, size_t id){
	if(!nn)
		return;
	struct layer *l = &nn->layers[id];

	if(!l)
		return;

	printf("Printing layer %i\nShape: %i->%i\nBias:\n", (int) id, (int)l->in, (int)l->out);
	print_vector(l->bias, l->out);

	printf("Weights:\n");
	for(size_t o = 0; o < l->out;o++){
		print_vector(&l->weights[o * l->in], l->in);
	}

	printf("\nEnd.\n");
}
