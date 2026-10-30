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