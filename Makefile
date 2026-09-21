CC=gcc
PARAMS=-Wall -Wextra -fsanitize=address

nn: *.c 
	${CC} ${PARAMS} $^ -g -lm -o $@.o
	./$@.o
