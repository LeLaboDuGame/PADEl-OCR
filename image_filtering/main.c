#include "images.h"

int main(int argc, char *argv[]){
    if (argc < 2){
        return 1;
    }
    Image *img = image_load(argv[1]);
    if (!img){
        printf("Error : %s\n", IMG_GetError());
        return 1;
    }

    int t = otsu_treshold(img);
    printf("seuil = %d\n", t);
	image_rotate(img, 0, 255);
    Image *bin = treshold(img, t);
    if (!bin){
        free_image(img);
        return 1;
    }

    int status = 0;
    int *cols, *rows, ncols, nrows;
    if (detect_grid_lines(bin, &cols, &ncols, &rows, &nrows) != 0){
        printf("No grid detection\n");
        status = 1;
    } else {
        printf("%d vertical, %d horizontal\n", ncols, nrows);
        if (save_cells(img, cols, ncols, rows, nrows) != 0){
            printf("Error cells cutting\n");
            status = 1;
        }
        free(cols);
        free(rows);
    }

    free_image(bin);
    free_image(img);
    return status;
}
