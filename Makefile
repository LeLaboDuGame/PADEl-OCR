CC = gcc
PARAMS = -Wall -Wextra -fsanitize=address -O3 -march=native -ffast-math -g

NN_DIR = src/neural_network
GENERATED = xor ocr xor_model.bin ocr_model.bin xor_training_dataset.bin xor_test_dataset.bin ocr_train_dataset.bin ocr_test_dataset.bin

all:
	$(MAKE) -C $(NN_DIR) all

clean:
	$(MAKE) -C $(NN_DIR) clean
	rm -f $(GENERATED) *.o

.PHONY: all clean
