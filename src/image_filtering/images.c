#include "images.h"


Image *image_create(int w, int h){ //this function will create an image following the struct Image definition
    if(w<=0 || h<= 0){
        return NULL;
    }
    Image *new_image = malloc(sizeof(Image));
    if (!new_image){
        return NULL;
    }
    //size of the image (width and height)
    new_image->w = w;
    new_image->h = h;
    unsigned char *pixel_buffer = calloc((size_t)w*h, sizeof(unsigned char)); //we need unsigned char for pixels to get a value between 0 and 255
    if(!pixel_buffer){
        free(new_image);
        return NULL;
    }
    new_image->data = pixel_buffer;
    return new_image;
}

void free_image(Image *i){ //frees the image, pretty self-explanatory
    if (!i){
        return;
    }
    free(i->data);
    free(i);
    return;
}


unsigned char rgb_to_gray(Uint8 r, Uint8 g, Uint8 b){ //hte grayscale function from SDL2's TP 
    return (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

Image * suface_to_image(SDL_Surface *surface){ //makes an image from a SDL_Surface (we cannot really use a SDL_Surface for what we want, so we convert it in an Image)
    Image *new_image = image_create(surface->w,surface->h);
    if(!new_image){
        return NULL;
    }
    Uint8 *pixels = surface->pixels;
    //iterate over every pixels and apply a grayscale to level every color
    for(int y=0; y<surface->h; y++){
        for(int x = 0; x<surface->w; x++){
            Uint8 *p = pixels + y * surface->pitch + x*4; //we gotta get x*sizeof(int) which is x*4 to move from a RGBA pxl to another
            new_image->data[y*new_image->w+x] = rgb_to_gray(p[0],p[1],p[2]); //we only care about RGB, not about A so we discard p[3]
        }
    }
    return new_image;
}

Image *image_load(const char *path_to_image){ //loads a sdl surface to format it and then convert it to an image
    SDL_Surface *loaded = IMG_Load(path_to_image); 
    if(!loaded){
        return NULL;
    }
    SDL_Surface *surface = SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(loaded); //w e don't need loaded anymore, only needed it to format the sdl surface that will be an image
    if(!surface){
        return NULL;
    }
    Image *img = suface_to_image(surface);
    SDL_FreeSurface(surface);
    return img;
}

int image_save(const Image *img, const char *path){ //saves the grayscale image
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, img->w, img->h, 32, SDL_PIXELFORMAT_RGBA32);
    if (!surface){
        return -1;
    }
    Uint8 *pixels = surface->pixels;
    //iterate over every pixel from the image to put it in the surface
    for (int y = 0; y < img->h; y++){
        for (int x = 0; x < img->w; x++){
            Uint8 v = img->data[y * img->w + x];
            Uint8 *p = pixels + y * surface->pitch + x * 4;
            p[0] = p[1] = p[2] = v;
            p[3] = 255; //set A of RGBA to 255 bc we do not really care about opacity
        }
    }
    int res = IMG_SavePNG(surface, path);
    SDL_FreeSurface(surface);
    return res;
}



Image *image_rotate(const Image *src, double angle_deg, unsigned char fill){
    double rad = angle_deg * PI / 180.0;
    double c = cos(rad);
    double s = sin(rad);

    // Size of the bounding box of the rotated image.
    // The - 1e-9 stops tiny floating-point errors (cos(90°) is not exactly 0)
    // from adding a useless extra pixel through ceil.
    int new_w = (int)ceil(fabs(src->w * c) + fabs(src->h * s) - 1e-9);
    int new_h = (int)ceil(fabs(src->w * s) + fabs(src->h * c) - 1e-9);

    Image *out = image_create(new_w, new_h);
    if (!out){
        return NULL;
    }

    // Centers of both images, in pixel coordinates
    double cx_src = (src->w - 1) / 2.0;
    double cy_src = (src->h - 1) / 2.0;
    double cx_out = (new_w - 1) / 2.0;
    double cy_out = (new_h - 1) / 2.0;

    for (int y = 0; y < new_h; y++){
        for (int x = 0; x < new_w; x++){
            // Position of the output pixel relative to the output center
            double dx = x - cx_out;
            double dy = y - cy_out;

            // Inverse rotation: where does this output pixel come from in src?
            double sx = dx * c + dy * s + cx_src;
            double sy = -dx * s + dy * c + cy_src;

            // Nearest source pixel
            int ix = (int)floor(sx + 0.5);
            int iy = (int)floor(sy + 0.5);

            if (ix >= 0 && ix < src->w && iy >= 0 && iy < src->h){
                out->data[y * new_w + x] = src->data[iy * src->w + ix];
            } else {
                out->data[y * new_w + x] = fill;   // outside the original image
            }
        }
    }
    return out;
}
