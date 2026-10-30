#ifndef IMAGES_H
#define IMAGES_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <math.h>
#define PI 3.14159265358979323846


typedef struct{
    int w,h;
    unsigned char *data; 
} Image;

Image *image_create(int w, int h);
void free_image(Image *i);
unsigned char rgb_to_gray(Uint8 r, Uint8 g, Uint8 b);
Image * suface_to_image(SDL_Surface *surface);
Image *image_load(const char *path);
int image_save(const Image *img, const char *path);
Image *image_rotate(const Image *src, double angle_deg, unsigned char fill);
Image *image_crop(const Image *src, int x, int y, int w, int h);
Image *image_resize_28(const Image *src);
int save_cells(const Image *gray, const int *cols, int ncols, const int *rows, int nrows);
int *projection_verticale(const Image *bin, int *len);
int *projection_horizontale(const Image *bin, int x_start, int x_end, int *len);
int group_peaks(const int *projection, int len, int threshold, int *positions);
int max_value(const int *projection, int len);
int find_lines(const int *projection, int len, int **positions);
int detect_grid_lines(const Image *img, int **cols, int *ncols,int **rows, int *nrows);
Image *image_crop(const Image *src, int x, int y, int w, int h);
Image *image_resize_28(const Image *src);
int save_cells(const Image *gray, const int *cols, int ncols, const int *rows, int nrows);
void compute_histogram(const Image *src, int hist[256]);
int otsu_treshold(const Image *src);
Image *treshold(const Image *src, int t);




#endif
