CC = gcc
PARAMS = -Wall -Wextra -fsanitize=address -O3 -march=native -ffast-math -g

FILTER_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(FILTER_DIR)/../..)
INCLUDE := -I$(ROOT_DIR)/include
SDL_CFLAGS := $(shell pkg-config --cflags sdl2 SDL2_image)
SDL_LIBS := $(shell pkg-config --libs sdl2 SDL2_image)

SRC := $(FILTER_DIR)main.c $(FILTER_DIR)images.c $(FILTER_DIR)filters.c $(FILTER_DIR)grid.c $(FILTER_DIR)cells.c
HEADERS := $(ROOT_DIR)/include/images.h
TARGET := filter

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CC) $(PARAMS) $(SDL_CFLAGS) $(INCLUDE) $(SRC) -o $(TARGET) $(SDL_LIBS) -lm

clean:
	rm -f $(TARGET) $(FILTER_DIR)*.o

.PHONY: all clean
