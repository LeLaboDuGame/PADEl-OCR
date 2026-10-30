#ifndef SOLVER_H
#define SOLVER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_SIZE 200

typedef struct {
    int rows;
    int cols;
    char grid[MAX_SIZE][MAX_SIZE];
} Grid;

int load_grid(const char *filename, Grid *g);
int check(const Grid *g, const char *word, int row, int col, int d);
int solve(const Grid *g, const char *word, int out[4]);

#endif