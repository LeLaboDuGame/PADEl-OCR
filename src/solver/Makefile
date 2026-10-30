CC = gcc
CPPFLAGS = -I../include
CFLAGS = -Wall -Wextra -std=gnu99
TARGET = solver
OBJ = solver.o
HEADERS = ../include/solver.h

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $<

%.o: %.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

debug: CFLAGS += -g -fsanitize=address
debug: clean $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean debug
