#ifndef IMAGES_H
#define IMAGES_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

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

#endif