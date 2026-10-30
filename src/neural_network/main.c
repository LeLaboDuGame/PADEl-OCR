#include "nn.h"
#include "dataset.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifndef XOR
#define XOR 1
#endif

void sigmoid(float *Z, float *A, size_t n){
	for(size_t i = 0; i < n; i++) A[i] = 1/(1 + exp(-Z[i]));
}

void dsigmoid(float *dA, float *Z, float *A, float *dZ, size_t n){
	(void) Z;
	for(size_t i = 0; i < n; i++){
		dZ[i] = dA[i]*A[i]*(1-A[i]);
	}
}

float ce(float *Ypred, float *T, size_t n){
	float s=0;
	float max = Ypred[0];
	for(size_t i = 1; i < n; i ++) max = fmaxf(max, Ypred[i]);
	for(size_t i = 0; i < n; i++) s += expf(Ypred[i] - max);
	s = logf(s);
	float target_logits = Ypred[0];
	for(size_t i = 0; i < n; i++){
		if(T[i] >= 0.5){
			target_logits = Ypred[i];
			break;
		}
	}
	return s - (target_logits - max);
}

void dce(float *Ypred, float *T, float *dA, size_t n){
	float max = Ypred[0];
	for(size_t i = 1; i < n; i++) max = fmaxf(max, Ypred[i]);

	float sum = 0.0f;
	for(size_t i = 0; i < n; i++) sum += expf(Ypred[i] - max);

	for(size_t i = 0; i < n; i++){
		float p = expf(Ypred[i] - max) / sum; // vraie proba softmax
		dA[i] = p - T[i];
	}
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

void lrelu(float *Z, float *A, size_t n){
	for(size_t i = 0; i < n; i++) A[i] = fmaxf(0.001*Z[i], Z[i]);
}	

void dlrelu(float *dA, float *Z, float *A, float *dZ, size_t n){
	(void)A;
	for(size_t i = 0; i < n; i++){
		dZ[i] = dA[i] * (Z[i] > 0 ? 1 : 0.001);
	}
}

void softmax(float *Z, float *A, size_t n){
	(void) Z;
	(void) A;
	(void) n;
	/*
	float max = Z[0];
	for(size_t i = 1; i < n; i++){
		Z[i] = log(fmaxf(Z[i], 1e-4));
		max = fmaxf(Z[i], max);
	}
	float sum = 0.0f;
    for(size_t j = 0; j < n; j++){
        sum += expf(Z[j] - max);
    }
	
	for(size_t i = 0; i < n; i++) {
        A[i] = expf(Z[i] - max) / sum;
	}
	*/
	// Simply identity as Softmax+CE vanish we need to use a special form of the mathematical function
	for(size_t i = 0; i < n; i++){
		A[i] = Z[i];
	}
	
}

void dsoftmax(float *dA, float *Z, float *A, float *dZ, size_t n){
	(void) Z;
	(void) A;
	/*
	(void) Z;
	float dot_product = 0.0f;
   
    for (size_t j = 0; j < n; j++) {
        dot_product += dA[j] * A[j];
    }
    for (size_t i = 0; i < n; i++) {
        dZ[i] = A[i] * (dA[i] - dot_product);
    }
    */
    for(size_t i = 0; i < n; i++){
    	dZ[i] = dA[i];
    }
    
}

#if XOR == 1
static int create_xor_dataset(char *filename){
	float a1[2] = {0, 0};
	float a2[2] = {0, 1};
	float a3[2] = {1, 0};
	float a4[2] = {1, 1};
	float *A[4] = {a1, a2, a3, a4};

	float o0[1] = {0};
	float o1[1] = {1};
	float *O[4] = {o0, o1, o1, o0};

	return create_dataset(4, 2, 1, filename, A, O);
}
#endif

#if XOR == 0
static int max_index(float *X, size_t n){
	float max = X[0];
	int id = 0;
	for(size_t i = 1; i < n; i++){
		if(X[i] > max){
			max = X[i];
			id = i;
		}
	}
	return id;
}
#endif

int main(){
	srand(time(NULL)); // Reset the randomisation seed

#if XOR == 1
	#define n_layers 3
	size_t layers[n_layers] = {2, 4, 1};
	Activation activations[n_layers - 1] = {{sigmoid, dsigmoid}, {sigmoid, dsigmoid}};
	Loss loss = {mse, dmse};

	char model_path[] = "xor_model.bin";
	char train_path[] = "xor_training_dataset.bin";
	char test_path[] = "xor_test_dataset.bin";

	printf("Loading model...\n");
	struct neural_network *nn = load_model(model_path, loss, activations);
	if(!nn){
		printf("Creating a new network\n");
		nn = create_nn(layers, activations, loss, n_layers);
	}

	if(!nn){
		printf("Neural Network couldn't be created: Malloc Error\n");
		return -1;
	}

	Dataset *d = load_dataset(train_path, 2, 1);
	if(!d){
		printf("Creating training dataset: %s\n", train_path);
		if(create_xor_dataset(train_path) == -1){
			printf("Dataset couldn't be created!\n");
			destroy_nn(nn);
			return -1;
		}
		d = load_dataset(train_path, 2, 1);
		if(!d){
			printf("Couldn't load training dataset: %s\n", train_path);
			destroy_nn(nn);
			return -1;
		}
	}

	Dataset *dt = load_dataset(test_path, 2, 1);
	if(!dt){
		printf("Creating test dataset: %s\n", test_path);
		if(create_xor_dataset(test_path) == -1){
			printf("Dataset couldn't be created!\n");
			destroy_dataset(d);
			destroy_nn(nn);
			return -1;
		}
		dt = load_dataset(test_path, 2, 1);
		if(!dt){
			printf("Couldn't load test dataset: %s\n", test_path);
			destroy_dataset(d);
			destroy_nn(nn);
			return -1;
		}
	}

	Cache *caches = create_caches(nn);
	if(!caches){
		printf("Cache couldn't be created: Malloc Error\n");
		destroy_dataset(d);
		destroy_dataset(dt);
		destroy_nn(nn);
		return -1;
	}

	Grad *grads = create_grads(nn);
	if(!grads){
		printf("Gradients couldn't be created: Malloc Error\n");
		destroy_caches(caches, nn->len + 1);
		destroy_dataset(d);
		destroy_dataset(dt);
		destroy_nn(nn);
		return -1;
	}

	train(nn, caches, grads, d->input_data, d->output_data, d->dataset_length, 5000, 1e-1);

	printf("XOR results:\n");
	float xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	for(size_t i = 0; i < 4; i++){
		forward(nn, xor_inputs[i], caches);
		printf("  %g XOR %g = %.6f\n", xor_inputs[i][0], xor_inputs[i][1], caches[nn->len].A[0]);
	}

	size_t good = 0, total = dt->dataset_length;
	for(size_t i = 0; i < total; i++){
		forward(nn, dt->input_data[i], caches);
		int need = dt->output_data[i][0] >= 0.5f;
		int got = caches[nn->len].A[0] >= 0.5f;
		if(got == need) good++;
	}
	printf("Accuracy: %zu/%zu = %.2f%%\n", good, total, total ? 100.0*good/total : 0.0);

	printf("Saving model...\n");
	save_model(nn, model_path);
	printf("Model saved!\n");

	destroy_grads(grads, nn->len + 1);
	destroy_caches(caches, nn->len + 1);
	destroy_dataset(d);
	destroy_dataset(dt);
	destroy_nn(nn);
#else
	#define n_layers 4
	size_t layers[n_layers] = {28*28, 128, 64, 26};
	Activation activations[n_layers - 1] = {{lrelu, dlrelu}, {lrelu, dlrelu}, {softmax, dsoftmax}};
	Loss loss = {ce, dce};

	char model_path[] = "ocr_model.bin";
	char train_path[] = "ocr_train_dataset.bin";
	char test_path[] = "ocr_test_dataset.bin";

	printf("Loading model...\n");
	struct neural_network *nn = load_model(model_path, loss, activations);
	if(!nn){
		printf("Creating a new network\n");
		nn = create_nn(layers, activations, loss, n_layers);
	}

	if(!nn){
		printf("Neural Network couldn't be created: Malloc Error\n");
		return -1;
	}

	Dataset *d = load_dataset(train_path, 28*28, 26);
	if(!d){
		printf("Couldn't load training dataset: %s\n", train_path);
		destroy_nn(nn);
		return -1;
	}

	Dataset *dt = load_dataset(test_path, 28*28, 26);
	if(!dt){
		printf("Couldn't load test dataset: %s\n", test_path);
		destroy_dataset(d);
		destroy_nn(nn);
		return -1;
	}

	Cache *caches = create_caches(nn);
	if(!caches){
		printf("Cache couldn't be created: Malloc Error\n");
		destroy_dataset(d);
		destroy_dataset(dt);
		destroy_nn(nn);
		return -1;
	}

	Grad *grads = create_grads(nn);
	if(!grads){
		printf("Gradients couldn't be created: Malloc Error\n");
		destroy_caches(caches, nn->len + 1);
		destroy_dataset(d);
		destroy_dataset(dt);
		destroy_nn(nn);
		return -1;
	}

	train(nn, caches, grads, d->input_data, d->output_data, d->dataset_length, 10, 5e-3);

	size_t good = 0, total = dt->dataset_length;
	for(size_t i = 0; i < total; i++){
		forward(nn, dt->input_data[i], caches);
		int need = max_index(dt->output_data[i], 26);
		int got = max_index(caches[nn->len].A, 26);
		if(got == need) good++;
	}
	printf("Accuracy: %zu/%zu = %.2f%%\n", good, total, total ? 100.0*good/total : 0.0);

	printf("Saving model...\n");
	save_model(nn, model_path);
	printf("Model saved!\n");

	destroy_grads(grads, nn->len + 1);
	destroy_caches(caches, nn->len + 1);
	destroy_dataset(d);
	destroy_dataset(dt);
	destroy_nn(nn);
#endif

	return 0;
}
