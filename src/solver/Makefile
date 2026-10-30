CC = gcc
PARAMS = -Wall -Wextra -fsanitize=address -O3 -march=native -ffast-math -g

SOLVER_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(SOLVER_DIR)/../..)
INCLUDE := -I$(ROOT_DIR)/include

SRC := $(SOLVER_DIR)solver.c
HEADERS := $(ROOT_DIR)/include/solver.h

all: solver

solver: $(SRC) $(HEADERS)
	$(CC) $(PARAMS) $(INCLUDE) $(SRC) -o solver

clean:
	rm -f solver $(SOLVER_DIR)*.o

.PHONY: all solver clean
