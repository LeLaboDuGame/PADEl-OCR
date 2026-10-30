#include "images.h"

// for each column x, counts the dark pixels found in that column
//Columns crossed by a vertical grid line get a very high count
int *projection_verticale(const Image *img, int *len){
    int *projection = calloc(img->w, sizeof(int)); //Gotta use calloc to make sure every pixel of the projection is usually set to 0
    if (!projection){
        return NULL;
    }
    for (int y = 0; y < img->h; y++){
        for (int x = 0; x < img->w; x++){
            if (img->data[y * img->w + x] == 0){ //img->data[y * img->w + x] == 0 <==> a dark pixel
                projection[x]++;
            }
        }
    }
    *len = img->w;
    return projection;
}

//for each row y counts the dark pixels in that row but only between columns x_start and x_end 
int *projection_horizontale(const Image *img, int x_start, int x_end, int *len){
    int *projection = calloc(img->h, sizeof(int));
    if (!projection){
        return NULL;
    }
    for (int y = 0; y < img->h; y++){
        for (int x = x_start; x <= x_end; x++){
            if (img->data[y * img->w + x] == 0){
                projection[y]++;
            }
        }
    }
    *len = img->h;
    return projection;
}

//groups consecutive values of a projection that are >= treshold into runs and stores the middle index of each run in positions[]
int group_peaks(const int *projection, int len, int threshold, int *positions){
    int count = 0;
    int start = -1; //if -1, not inside a run; otherwise index where thje current run began

    for (int i = 0; i < len; i++){
        if (projection[i] >= threshold){
            if (start == -1){
                start = i; //new run here
            }
        } else if (start != -1){
            //the run ended at i-1 and we record its middle
            positions[count] = (start + i - 1) / 2;
            count++;
            start = -1;
        }
    }
    // A run that reaches the last index is never closed by the loop so we close it here
    if (start != -1){
        positions[count] = (start + len - 1) / 2;
        count++;
    }
    return count;
}

// returns the largest value in the projection
int max_value(const int *projection, int len){
    int a = 0;
    for (int i = 0; i < len; i++){
        if (projection[i] > a){
            a = projection[i];
        }
    }
    return a;
}

//Finds the grid lines from a projection using a threshold at 7.5/10 of the maximum
//It adapts to the image size: letter columns stay below it, line columns go above it.
int find_lines(const int *projection, int len, int **positions){
    *positions = NULL;
    int *pos = malloc(len * sizeof(int));
    if (!pos){
        return -1;
    }
    int threshold = max_value(projection, len) * 7.5 / 10;
    int n = group_peaks(projection, len, threshold, pos);
    *positions = pos;
    return n;
}

//function that uses all other functions to give the position of each col and each row and their count
int detect_grid_lines(const Image *img, int **cols, int *ncols,int **rows, int *nrows){
    int len;
    //vertical lines, using the whole image width
    int *proj_v = projection_verticale(img, &len);
    if (!proj_v){
        return -1;
    }
    int n = find_lines(proj_v, len, cols);
    free(proj_v);
    if (n < 2){ // fewer than 2 lines = no cell can exist
        free(*cols);
        return -1;
    }
    *ncols = n;
    //horizontal lines only on the grid width
    int *proj_h = projection_horizontale(img, (*cols)[0], (*cols)[n - 1], &len);
    if (!proj_h){
        free(*cols);
        return -1;
    }
    int m = find_lines(proj_h, len, rows);
    free(proj_h);
    if (m < 2){
        free(*cols);
        free(*rows);
        return -1;
    }
    *nrows = m;

    return 0;
}
