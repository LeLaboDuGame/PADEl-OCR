CC=gcc
PARAMS=-Wall -Wextra -fsanitize=address -O3 -march=native -ffast-math

nn: *.c 
	${CC} ${PARAMS} $^ -g -lm -o $@.o
	./$@.o


dataset: dataset.c
	${CC} ${PARAMS} $^ -g -lm -o $@.o
	./$@.o
