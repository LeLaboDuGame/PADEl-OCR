#ifndef CELLS_H
#define CELLS_H

#include "images.h"

Image *image_crop(const Image *src, int x, int y, int w, int h);
int save_cells(const Image *gray, const int *cols, int ncols, const int *rows, int nrows);




#endif 