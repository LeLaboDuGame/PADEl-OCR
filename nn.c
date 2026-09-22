#include "nn.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

// Verbose level
short VERBOSE = 2;


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
		fill_random_array(l->bias, layers[i + 1]);
		
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
				
		fill_random_array(l->weights, layers[i+1] * layers[i]);
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


	// Output
	nn->loss.backward(Ypred, T, dA_last, L->out);
	L->activation.backward(dA_last, Z_last, dZ_last, L->out);

	// Update weaights and bias
	for(size_t o = 0; o < L->out; o++){
		L->bias[o] -= lr * dZ_last[o];
		for(size_t i = 0; i < L->in; i++){
			L->weights[o*L->in+i] -= lr * dZ_last[o] * caches[nn->len-1].A[i];
		}
	}

	// Hiden propagation
	for(int l = (int)nn->len - 2; l >= 0; l--){
		L = &nn->layers[l];
		struct layer *Lnext = &nn->layers[l+1];
		// dA[l+1] = W_next^T * dZ_next
		for(size_t i = 0; i < Lnext->in; i++){
			float sum = 0;
			for (size_t k = 0; k < Lnext->out; k++){
				sum += Lnext->weights[k*Lnext->in + i] * grads[l+2].dZ[k];
			}
			grads[l+1].dA[i] = sum;
		}
		// dZ = f'(Z)*dA
		L->activation.backward(grads[l + 1].dA, caches[l+1].Z, grads[l+1].dZ, L->out);

		// Update weights and bias
		for(size_t o = 0; o < L->out; o++){
			L->bias[o] -= lr * grads[l+1].dZ[o];
			for(size_t i = 0; i < L->in; i++){
				L->weights[o*L->in+i] -= lr * grads[l+1].dZ[o] * caches[l].A[i];
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
			float mean_loss = loss_sum / N;
			printf("Epochs n°%i/%i - Loss: %f\n", (int)e, (int)epochs, mean_loss);
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


// ------------ TESTING -------------------


void f(float *Z, float *A, size_t n){
	for(size_t i = 0; i < n; i++) A[i] = 1/(1 + exp(-Z[i]));
}

void df(float *dA, float *Z, float *dZ, size_t n){
	for(size_t i = 0; i < n; i++){
		float s = 1/(1+exp(-Z[i]));
		dZ[i] = dA[i]*s*(1-s);
	}
}

float ce(float *Ypred, float *T, size_t n){
	float s=0;
	for(size_t i = 0; i < n; i++) s += -T[i]*log(Ypred[i]+1e-8);
	return s;
}

void dce(float *Ypred, float *T, float *dA, size_t n){
 	for(size_t i = 0; i < n; i++) dA[i] = -T[i]/(Ypred[i]+1e-8);
}


float mse(float *Ypred, float *T, size_t n){
	float sum = 0;
	for(size_t i = 0; i < n; i++){
		sum += (T[i] - Ypred[i]) * (T[i] - Ypred[i]);
	}
	return 1 / (float)n * sum;
}

void dmse(float *Ypred, float *T, float *dA, size_t n){
	for(size_t i = 0; i < n; i++){
		dA[i] = 2/(float)n*(Ypred[i] - T[i]);
	}
}


int main(){ 
	srand(time(NULL)); // Reset the randomisation seed
	#define n_layers 6
	size_t layers[n_layers] = {2, 16, 32, 16, 8, 1};
	Activation activations[n_layers - 1] = {{f, df}, {f, df}, {f, df}, {f, df}, {f, df}};
	Loss loss = {mse, dmse};

	struct neural_network *nn = create_nn(layers, activations, loss, n_layers);
	//struct neural_network *nn = load_model("model.save", loss, activations);
	if(!nn){
		printf("Neural Network couldn't be created: Malloc Error\n");
		return -1;
	}

	Cache *caches = create_caches(nn);
	if(!caches){
		printf("Cache couldn't be created: Malloc Error\n");
		destroy_nn(nn);
		return -1;
	}

	/// Testing the save and load model functions
	float Xt[2]= {0, 0};

	forward(nn, Xt, caches);
	print_vector(caches[n_layers - 1].A, 1);
	nn_summary(nn);

	printf("Saving model\n");
	save_model(nn, "model.save");
	printf("Model saved!\n");
	destroy_nn(nn);

	printf("Loading model...\n");
	nn = load_model("model.save", loss, activations);
	printf("Model loaded!\n");
	forward(nn, Xt, caches);
	print_vector(caches[n_layers - 1].A, 1);
	nn_summary(nn);

	/*
	Grad *grads = create_grads(nn);
	if(!grads){	
		printf("Gradients could'nt be created: Malloc Error\n");
		destroy_caches(caches, n_layers);
		destroy_nn(nn);
		return -1;
	}

	/// This trainng exemple return the gap between two number
	/// Exemple:
	/// 	{1, 0} gap = 1
	/// 	{0.3, 1} gap = 0.7
	/// 	{0.5, 0.5} gapt = 0 etc
	
	#define training_set 100
	float Xs_data[training_set][2];
	float Ts_data[training_set][1];
	float *Xs[training_set] = {0}; 
	float *Ts[training_set] = {0};

	// Generate random datasets
	for(size_t i = 0; i < training_set; i++){
		Xs_data[i][0] = random1m1()/2;
		Xs_data[i][1] = random1m1()/2;
		Xs[i] = Xs_data[i];
		Ts_data[i][0] = fabsf(Xs_data[i][0] - Xs_data[i][1]);
		Ts[i] = Ts_data[i];
	}
	

	train(nn, caches, grads, Xs, Ts, training_set, 10, 1e-1 * 2);

	#define n_test 9
	float Xtest[n_test][2] = {{0, 0}, {0.5, 0.5}, {1, 1}, {0.5, 1}, {1, 0.5}, {0.2, 0.5}, {0.8, 1}, {1, 0}, {0, 1}};

	for(size_t i = 0; i < n_test; i++){
		forward(nn, Xtest[i], caches);
		printf("|Test n°%i - for: ", (int)i);
		print_vector(Xtest[i], 2);
		printf(" - got: %f|\n", caches[nn->len].A[0]);
	}
	*/
	/*
	printf("Starting forward propagation...\n");
	float X[2] = {1};
	float T[1] = {1};
	forward(nn, X, caches);

	printf("Previous loss: %f\n", loss.forward(caches[2].A, T, 1));
	
	printf("Forward propagation finished!\nPrinting result:\n\n");
	for(int i = 0; i < n_layers; i++){
		printf("Layer n°%i:\n", i);
		print_vector(caches[i].A, layers[i]);
	}

	
	float loss_val = nn->loss.forward(caches[2].A, T, 1);
	nn->loss.backward(caches[2].A, T, grads[2].dA, 1);
	printf("Testing loss+dA:\nloss: %f\ndA: \n", loss_val);
	print_vector(grads[2].dA, 1);
	
	printf("\nStarting backpropagation...\n");
	
	backward(nn, caches, grads, T, 0.1);
	printf("Backpropagation finished!\n");

	forward(nn, X, caches);
	printf("New loss: %f\n", loss.forward(caches[2].A, T, 1));
	
	destroy_grads(grads, n_layers);
	*/
	destroy_caches(caches, n_layers);
	destroy_nn(nn);
	return 0;
}
