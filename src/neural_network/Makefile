CC = gcc
PARAMS = -Wall -Wextra -fsanitize=address -O3 -march=native -ffast-math -g

NN_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(NN_DIR)/../..)
INCLUDE := -I$(ROOT_DIR)/include

SRC := $(NN_DIR)main.c $(NN_DIR)nn.c $(NN_DIR)utils.c $(ROOT_DIR)/src/dataset/dataset.c
HEADERS := $(ROOT_DIR)/include/nn.h $(ROOT_DIR)/include/dataset.h $(ROOT_DIR)/include/utils.h

all: xor ocr

xor: $(SRC) $(HEADERS)
	$(CC) $(PARAMS) $(INCLUDE) -DXOR=1 $(SRC) -lm -o xor

ocr: $(SRC) $(HEADERS)
	$(CC) $(PARAMS) $(INCLUDE) -DXOR=0 $(SRC) -lm -o ocr

GENERATED = xor ocr xor_model.bin ocr_model.bin xor_training_dataset.bin xor_test_dataset.bin ocr_train_dataset.bin ocr_test_dataset.bin

clean:
	rm -f $(GENERATED) $(NN_DIR)*.o

.PHONY: all xor ocr clean
