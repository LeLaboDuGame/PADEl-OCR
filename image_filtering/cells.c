#include "cells.h"

//returns the cropped image of the w*h rectangle of src whose top left corner is x,y
Image *image_crop(const Image *src, int x, int y, int w, int h){
    //reject empty rectangles and rectangles that stick out of the source image
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > src->w || y + h > src->h){
        return NULL;
    }
    Image *out = image_create(w, h);
    if (!out){
        return NULL;
    }
    for (int j = 0; j < h; j++){
        for (int i = 0; i < w; i++){
            //pixel i,j of the output is pixel x+i,y+j of the source
            out->data[j * w + i] = src->data[(y + j) * src->w + (x + i)];
        }
    }
    return out;
}

//cuts every cell of the grid out of the grayscale image and saves each one as cells/cell_(rows)_(cols).png
int save_cells(const Image *gray, const int *cols, int ncols, const int *rows, int nrows){
    //cols and rows are the grid line positions found by detect_grid_lines so the grid has (ncols - 1) x (nrows - 1) cells
    //the margin is arbitrary but fits the first two images so its fine for now. It's based on the first cell's width and is applied to alol
    int margin = (cols[1] - cols[0]) / 8;

    for (int i = 0; i < nrows - 1; i++){
        for (int j = 0; j < ncols - 1; j++){
            // the cell lies between lines j and j+1 horizontally and i and i+1 vertically and all are shrunk by the margin on each side
            int x = cols[j] + margin;
            int y = rows[i] + margin;
            int w = cols[j + 1] - cols[j] - 2 * margin;
            int h = rows[i + 1] - rows[i] - 2 * margin;

            Image *cell = image_crop(gray, x, y, w, h);
            if (!cell){
                return -1;
            }

            char name[64];
            snprintf(name, sizeof(name), "cells/cell_%02d_%02d.png", i, j);
            image_save(cell, name);
            free_image(cell); // the crop is no longer needed once saved
        }
    }
    return 0;
}