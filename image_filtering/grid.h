#ifndef GRID_H
#define GRID_H
#include "images.h"

int *projection_verticale(const Image *bin, int *len);
int *projection_horizontale(const Image *bin, int x_start, int x_end, int *len);
int group_peaks(const int *projection, int len, int threshold, int *positions);
int max_value(const int *projection, int len);
int find_lines(const int *projection, int len, int **positions);
int detect_grid_lines(const Image *img, int **cols, int *ncols,int **rows, int *nrows);


#endif