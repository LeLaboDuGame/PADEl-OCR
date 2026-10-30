#include "images.h"

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

Image *image_resize_28(const Image *src) {
    int target_w = 28;
    int target_h = 28;
    
    Image *out = image_create(target_w, target_h);
    if (!out) {
        return NULL;
    }

    float scale_x = (float)src->w / target_w;
    float scale_y = (float)src->h / target_h;

    for (int j = 0; j < target_h; j++) {
        for (int i = 0; i < target_w; i++) {
            // Position of the src img
            float gx = i * scale_x;
            float gy = j * scale_y;

            // Coordonnates of the 4 pixels around
            int gxi = (int)gx;
            int gyi = (int)gy;

            // interpolation wheight computation
            float tx = gx - gxi;
            float ty = gy - gyi;

            // src img edge handling 
            int next_x = (gxi + 1 < src->w) ? gxi + 1 : gxi;
            int next_y = (gyi + 1 < src->h) ? gyi + 1 : gyi;

            // Gets the 4 pixels back
            unsigned char c00 = src->data[gyi * src->w + gxi];
            unsigned char c10 = src->data[gyi * src->w + next_x];
            unsigned char c01 = src->data[next_y * src->w + gxi];
            unsigned char c11 = src->data[next_y * src->w + next_x];

            // bilinear interpolation 
            float top = c00 + tx * (c10 - c00);
            float bottom = c01 + tx * (c11 - c01);
            float value = top + ty * (bottom - top);

            // Fills the pixels in out
            out->data[j * target_w + i] = (unsigned char)value;
        }
    }

    return out;
}

//cuts every cell of the grid out of the grayscale image and saves each one as cells/cell_(rows)_(cols).png
int save_cells(const Image *gray, const int *cols, int ncols, const int *rows, int nrows){
    //cols and rows are the grid line positions found by detect_grid_lines so the grid has (ncols - 1) x (nrows - 1) cells
    //the margin is arbitrary but fits the first two images so its fine for now. It's based on the first cell's width and is applied to all
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

			Image *resized_cell = image_resize_28(cell);
			free_image(cell);

			if (!resized_cell){
			    return -1;
			}
			
            char name[64];
            snprintf(name, sizeof(name), "cells/cell_%02d_%02d.png", i, j);
            image_save(resized_cell, name);
            free_image(resized_cell); // the crop is no longer needed once saved
        }
    }
    return 0;
}
