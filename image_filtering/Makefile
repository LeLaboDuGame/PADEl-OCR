CC = gcc
CFLAGS = -Wall -Wextra -g -I../../include $(shell pkg-config --cflags sdl2 SDL2_image)
LDLIBS = $(shell pkg-config --libs sdl2 SDL2_image) -lm

TARGET = ocr

SRCS = main.c images.c filters.c grid.c cells.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.c images.h filters.h grid.h cells.h
images.o: images.c images.h
filters.o: filters.c filters.h images.h
grid.o: grid.c grid.h images.h
cells.o: cells.c cells.h images.h

clean:
	rm -f $(OBJS)

mrproper: clean
	rm -f $(TARGET)

.PHONY: all clean mrproper
